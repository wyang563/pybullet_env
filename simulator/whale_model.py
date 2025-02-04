import socket
import threading
import pybullet as p
import cv2
import numpy as np
from simulator.simulator_utils import *
import csv
from simulator.icp import icp

# IMPORTANT CONSTANTS FOR RECON SEARCH
SEARCH_SPEED = 0.25
DEFAULT_SPEED = 0.12
WHALE_TRACK_SPEED = 0.05
VERT_TIME = 2200
TURN_TIME = 4800
LANDING_SPEED = -0.35
LANDING_HEIGHT = 0.5
CRUISE_HEIGHT = 4
IN_POSITION_DIST = 1.1 # distance tracking drones to search drone to be considered in position
CTRL_DIST = 0.25

class WhaleDroneModel:
    def __init__(self, drone_id, env, sim_dir, num_drones, lead_drone=False):
        self.drone_id = drone_id # IMPORTANT: drone_id is the index of the drone in our drones list
        self.mode = "search"
        self.lead_drone = lead_drone
        self.num_drones = num_drones
        self.env = env
        self.turning = False
        self.turn_target_yaw = None
        self.in_command = None # formatted as "command type | command text"
        self.sim_dir = sim_dir
        self.debug_file = f"{sim_dir}/debug/drone_{self.drone_id}.csv"
        self.timestep = -1
        self.target_obj_index = -1 # uninitialized at start

    def set_other_drones(self, other_drones):
        # map of ids to drone objects 
        self.other_drones = other_drones
    
    def set_timestep(self):
        self.timestep += 1
    
    def get_drone_state(self):
        return self.env._getDroneStateVector(int(self.drone_id))

    def get_turn_yaw_rate(self, timestep=None):
        YAW_STEP_RATE = 20
        cur_yaw = self.env._getDroneStateVector(int(self.drone_id))[9]
        if abs(self.turn_target_yaw - cur_yaw) < 0.05:
            print("FINISHED TURNING to target yaw: ", self.turn_target_yaw, "time: ", timestep)
            self.turning = False
            return 0
        return (self.turn_target_yaw - cur_yaw) / YAW_STEP_RATE
    
    def rotate_velocity(self, velocity):
        velocity = np.array([velocity])
        drone_yaw = self.env._getDroneStateVector(int(self.drone_id))[9]
        rot_matrix = np.array([[np.cos(drone_yaw), -np.sin(drone_yaw)], 
                               [np.sin(drone_yaw), np.cos(drone_yaw)]])
        rot_velocity = rot_matrix @ velocity.T
        return list(rot_velocity.T[0])

    ## Image Processing/Position Update 

    def check_color(self, pixel):
        '''
        Checks if pixel color is either blue or green
        '''
        # green
        if pixel[1] > 150 and pixel[0] < 60 and pixel[2] < 60:
            return True
        # blue
        elif pixel[2] > 150 and pixel[0] < 60 and pixel[1] < 60:
            return True
        return False


    def segment_image(self, seg):
        '''
        Return global coordinate of centers of clusters in segmented image
        and draw oriented bounding boxes around each segmented object.
        '''
        num_labels, labels_im = cv2.connectedComponents(seg.astype(np.uint8))
        centers = []
        boxes = []

        for label in range(1, num_labels):  # Skip label 0 (background)
            # Find pixels belonging to the current label
            object_pixels = np.argwhere(labels_im == label)

            # Calculate the center point
            center_x, center_y = object_pixels.mean(axis=0)
            centers.append((center_x, center_y))

            rect = cv2.minAreaRect(object_pixels[:, ::-1])  
            box = cv2.boxPoints(rect) 
            box = np.int0(box)  # Convert to integer coordinates
            boxes.append(box)

        # Generate a color mapping for each component
        if np.max(labels_im) > 0:
            label_hue = np.uint8(179 * labels_im / np.max(labels_im))  
            blank_ch = 255 * np.ones_like(label_hue)
            colored_img = cv2.merge([label_hue, blank_ch, blank_ch])
            colored_img = cv2.cvtColor(colored_img, cv2.COLOR_HSV2BGR)

            # Set the background label (0) to black
            colored_img[label_hue == 0] = (0, 0, 0)

            # Display the image
            plt.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
            plt.title('Connected Components with Oriented Bounding Boxes')
            plt.axis('off')

            # Save the colored connected components image with bounding boxes
            if self.timestep % 400 == 0:
                cv2.imwrite(self.sim_dir + f"/segment_pics{self.drone_id}/segments_drone_{self.timestep}.png", colored_img)
            
            for box in boxes:
                cv2.drawContours(colored_img, [box], 0, (0, 255, 0), 2)  # Draw the rectangle in green

            if self.timestep % 400 == 0:
                cv2.imwrite(self.sim_dir + f"/center_pics{self.drone_id}/segments_drone_{self.timestep}.png", colored_img) 
        return centers, boxes

    def check_whales(self, seg, rgb):
        '''
        Given a drone shot image at a given time step, segment the image and find global positions of all objects in
        image. Calculates velocity drone needs to fly to reach object. 
        '''

        # find center points of segmented image
        centers, _ = self.segment_image(seg)
        count = 0
        for x, y in centers:
            count += int(self.check_color(rgb[int(x), int(y)]))
        return count >= self.num_drones - 1
    
    def pixel_to_world_velocity(self, pixel_coord, img_dims):
        '''
        Drone is at center of image, get velocity to target pixel coordinate
        '''
        m, n = img_dims[0], img_dims[1]
        x, y = pixel_coord
        dx, dy = -x + m/2, y - n/2
        incr_constant = np.sqrt(WHALE_TRACK_SPEED ** 2 / (dx ** 2 + dy ** 2))
        self.write_debug_file([x, y, dx, dy])
        # print("target vector: ", [dx * incr_constant, dy * incr_constant])
        return [dy * incr_constant, dx * incr_constant, 0, 0]

    def get_whale_center_list(self, seg, rgb):
        '''
        returns pixel coordinate list of whale centers along with their corresponding bounding boxes
        '''
        centers, boxes = self.segment_image(seg)
        whale_centers = []
        whale_boxes = []
        for i, (x, y) in enumerate(centers):
            if self.check_color(rgb[int(x), int(y)]):
                whale_centers.append((x, y))
                whale_boxes.append(boxes[i])
        return whale_centers, whale_boxes

    def get_whales_center(self, seg, rgb):
        '''
        get whale center and calculate velocity to rough area where whale center is
        '''
        centers, _ = self.segment_image(seg)
        avg_x, avg_y = 0, 0
        count = 0
        for x, y in centers:
            if self.check_color(rgb[int(x), int(y)]):
                avg_x += x
                avg_y += y
                count += 1
        if count == 0:
            return [0, 0, 0, 0]
        else:
            target_point = [avg_x / count, avg_y / count]
            return self.pixel_to_world_velocity(target_point, seg.shape)
    
    # calculate distances to other drones, returns yes if dist is too close to another drone
    def check_drone_proximity(self):
        for drone in self.other_drones:
            if drone != self.drone_id:
                drone_state = self.other_drones[drone].get_drone_state()
                dist = np.linalg.norm(np.array(drone_state[:3]) - np.array(self.get_drone_state()[:3]))
                if dist < CTRL_DIST:
                    return True
        return False

    # tracking stage functions
    def track_whale(self, seg, rgb):
        centers, _ = self.segment_image(seg)
        whale_centers = []
        for x, y in centers:
            if self.check_color(rgb[int(x), int(y)]):
                whale_centers.append((y, x))
        
        whale_centers.sort()
        try:
            target_center = whale_centers[int(self.drone_id) - 1]
            # check if we're centered on whale
            target_center = [target_center[1], target_center[0]]
            if (target_center[0] - seg.shape[0] // 2)**2 + (target_center[1] - seg.shape[1] // 2)**2 < 10:
                return [0, 0, 0, 0], True
            return self.pixel_to_world_velocity(target_center, seg.shape), False
        except:
            print("target center error")
            return [0, 0, 0, 0], False

    # landing logic
    def land_drone(self):
        if self.get_drone_state()[2] < LANDING_HEIGHT:
            return [0, 0, 0, 0], True
        return [0, 0, LANDING_SPEED, 0], False
    
    # receiving server side commands
    def calc_velocity_to_point(self, target_point):
        x, y = self.get_drone_state()[:2]
        dx, dy = target_point[0] - x, target_point[1] - y
        # calculate velocity to reach target point
        incr_constant = np.sqrt(DEFAULT_SPEED ** 2 / (dx ** 2 + dy ** 2))
        return [dx * incr_constant, dy * incr_constant, 0, 0]
        # rot_velocity = self.rotate_velocity([dx * incr_constant, dy * incr_constant])

    def receive_command(self):
        # check if we aren't too close to another drone, only freeze if we are drone 2
        if self.check_drone_proximity() and self.drone_id == "2":
            return [0, 0, 0, 0]

        if self.in_command is not None:
            command_type, text = self.in_command.split('|')
            if command_type == "velocity":
                x, y = text.split(',')
                return [float(x), float(y), 0, 0]
            
            elif command_type == "fly-to":
                x, y = text.split(',')
                return self.calc_velocity_to_point([float(x), float(y)])
        
        # reset input channel
        self.in_command = None

    # accessing image files
    def get_latest_images(self, img_type, last_n=1):
        if img_type == "rgb":
            file_dir = f"{self.sim_dir}/pics{self.drone_id}_track"
        elif img_type == "seg":
            file_dir = f"{self.sim_dir}/segment_pics{self.drone_id}"
        else:
            raise ValueError("Invalid image type")
        result = []
        for img in sorted(os.listdir(file_dir))[-last_n:]:
            result.append(cv2.imread(f"{file_dir}/{img}"))
        return result
    
    # debugging
    def write_debug_file(self, debug_strs):
        with open(self.debug_file, 'a') as f:
            debug_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            debug_writer.writerow([*debug_strs])

    # plot centers on rgb image and save it 
    def plot_centers_on_rgb(self, centers, rgb, drone_num):
        rgb_with_centers = rgb.copy()
        dot_color = (0, 0, 0)         
        radius = 2        
        thickness = -1
        
        # Iterate over each center and draw a dot on the image
        for center in centers:
            center = tuple(map(int, center))
            cv2.circle(rgb_with_centers, center, radius, dot_color, thickness)
        
        cv2.imwrite(self.sim_dir + f"/center_pics{drone_num}/centers_{self.timestep}.png", rgb_with_centers)
        return rgb_with_centers

class WhaleDroneLeadModel(WhaleDroneModel):
    '''Lead Drone model'''
    def __init__(self, drone_id, env, sim_dir, num_drones, lead_drone, init_position):
        super().__init__(drone_id=drone_id, env=env, num_drones=num_drones, sim_dir=sim_dir, lead_drone=lead_drone)
        self.search_state = 0 # 0: flying (1, 1), 1: flying -x direction, 2: flying x direction, 3: flying vertically, 4: turning
        self.start_vertical_timestep = 0 # timestep when drone starts moving vertically
        self.start_turning_timestep = 0
        self.target_y = 0
        self.init_velocity = [(9.0 - init_position[0]) / 24, (9 - init_position[1]) / 24, 0, 0]
        self.search_target_points = [] # (dx, dy) for the positions that each tagging drone should be at relative to search drone before tracking commences
        print("LEAD DRONE INIT VELOCITY IS: ", self.init_velocity)

    def stop_turn(self, timestep):
        # adjust currrent yaw to be 0
        cur_yaw = self.env._getDroneStateVector(int(self.drone_id))[9]

        if timestep - self.start_turning_timestep > TURN_TIME:
            self.turning = False
            # we do this so we can start 
            if self.search_state == 3:
                self.start_vertical_timestep = timestep
            print("FINISHED TURNING: ", timestep)
        return (-cur_yaw) / 20
    
    def search_step(self, timestep):
        '''
        North Edge: -0.03399542849160775,9.451969159352018,3.998695341601934
        East edge: 9.856670187306824,0.7086563736845014,3.998695341601934
        South edge: -10, -0.47408566127094487,3.998695341601934
        West edge: 0.2607578133471447,-9.553665922974535,3.9986953416019335
        '''
        state = self.get_drone_state()
        x = state[0]

        if self.turning:
            yaw_adj = self.stop_turn(timestep)
            return [0, 0, 0, yaw_adj]
        
        if self.search_state == 0:
            self.turning = False
            if x > 9:
                print("FINISHED STATE 0: ", timestep)
                self.search_state = 1
                self.turning = True
                self.start_turning_timestep = timestep
                self.turn_target_yaw = np.pi / 2
            return self.init_velocity
        
        elif self.search_state == 1:
            self.turning = False
            if x < -9:
                print("FINISHED STATE 1: ", timestep)
                self.search_state = 3
                self.turning = True
                self.start_turning_timestep = timestep
                self.turn_target_yaw = np.pi
            return [-SEARCH_SPEED, 0, 0, 0]
        
        elif self.search_state == 2:
            self.turning = False
            if x > 9:
                print("FINISHED STATE 2: ", timestep)
                self.search_state = 3
                self.turning = True
                self.start_turning_timestep = timestep
                self.turn_target_yaw = np.pi
            return [SEARCH_SPEED, 0, 0, 0]
        
        else:
            if timestep - self.start_vertical_timestep > VERT_TIME:
                print("FINISHED STATE 3: ", timestep, "start timestep", self.start_vertical_timestep)
                if x > 9:
                    self.search_state = 1
                    self.turning = True
                    self.start_turning_timestep = timestep
                    self.turn_target_yaw = np.pi / 2
                else:
                    self.search_state = 2
                    self.turning = True
                    self.start_turning_timestep = timestep
                    self.turn_target_yaw = -np.pi / 2
            return [0, -SEARCH_SPEED, 0, 0]
        
    def calc_search_target_points(self):
        search_width = min(self.num_drones - 2, 2.5)
        dx = -search_width / 2
        dy = 0
        for _ in range(self.num_drones - 1):
            self.search_target_points.append((dx, dy))
            dx += search_width / (self.num_drones - 2)
    
    def all_drones_in_position(self):
        search_drone_pos = self.get_drone_state()[:2]
        for drone in self.other_drones:
            if drone != self.drone_id:
                drone_pos = self.other_drones[drone].get_drone_state()[:2]
                dx = self.search_target_points[int(drone) - 1][0]
                if np.linalg.norm(np.array(drone_pos) - np.array(search_drone_pos)) > abs(dx) + 0.2:
                    return False
                if self.other_drones[drone].mode != "whales":
                    return False
        return True
    
    def all_drones_landed(self):
        for drone in self.other_drones:
            if drone != self.drone_id:
                if self.other_drones[drone].mode != "complete":
                    return False
        return True

    def icp_analysis(self):
        all_correspondences = []
        net_corr = [i for i in range(self.num_drones)] # mapping from drone 1 positions to drone x positions aggregatively
        for d in range(1, self.num_drones):
            rgb1, _, seg1 = self.env._getDroneImages(d)
            drone1_height = float(self.other_drones[str(d)].get_drone_state()[2])

            if d + 1 == self.num_drones:
                rgb2, _, seg2 = self.env._getDroneImages(1)
                drone2_height = float(self.other_drones[str(1)].get_drone_state()[2])

            else:
                rgb2, _, seg2 = self.env._getDroneImages(d + 1)
                drone2_height = float(self.other_drones[str(d + 1)].get_drone_state()[2])

            # get centers of whales
            whale_centers1, whale_boxes1 = self.get_whale_center_list(seg1, rgb1)
            whale_centers2, whale_boxes2 = self.get_whale_center_list(seg2, rgb2)
            print(whale_boxes1)
            print(whale_boxes2)

            assert len(whale_centers1) == len(whale_centers2), f"Number of whales detected in images do not match centers1: {len(whale_centers1)}, centers2: {len(whale_centers2)}"
            
            # plot centers on image
            
            _, corr = icp(np.array(whale_centers1), np.array(whale_centers2))

            all_correspondences.append(corr)
            
            # calculate net correspondence
            new_correspondence = [0 for _ in range(self.num_drones)]
            for i in range(len(corr)):
                new_correspondence[i] = net_corr[corr[i]]
            net_corr = new_correspondence.copy()
        
        # assert the mapping is the identity at the end
        assert net_corr == [i for i in range(self.num_drones)], "Final correspondence mapping is not the identity"
        return all_correspondences

    # send command to specific drone
    def send_command_drone(self, command, text, target_drone):
        self.other_drones[target_drone].in_command = f"{command}|{text}"

    # generalized send_command function to all drones
    def send_command_all_drones(self, command, text):
        for drone in self.other_drones:
            if drone != self.drone_id:
                self.other_drones[drone].in_command = f"{command}|{text}"

    # for lead drone to send fly commands to all drones during whale tracking phase
    def track_stage_send_command(self, cur_pos):
        x, y = cur_pos[0], cur_pos[1]
        for i in range(1, self.num_drones):
            dx, dy = self.search_target_points[i - 1]
            self.send_command_drone("fly-to", f"{x + dx},{y + dy}", str(i))
    