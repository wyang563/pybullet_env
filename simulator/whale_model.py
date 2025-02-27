import socket
import threading
import pybullet as p
import cv2
import numpy as np
from simulator.simulator_utils import *
import csv
from simulator.icp import icp
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
import dgl
import torch
from .gnn.modelv2 import NonLinearModel

# IMPORTANT CONSTANTS FOR RECON SEARCH
SEARCH_SPEED = 0.25
DEFAULT_SPEED = 0.12
WHALE_TRACK_SPEED = 0.10
WHALE_CHECK_THRESHOLD = 75 # Number of iterations whale count is the same before switching to whales mode
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
        self.prev_target_pixel_pos = None # for tracking stage (when using ICP), previous assigned drone target pixel position
        self.num_whales = None # initialized in whale search stage once scout drone determines how many whales there are

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
        THIS IMPLEMENTATION IS ONLY FOR TRACKING DRONES - SCOUTING DRONE IMPLEMENTATION BELOW
        Given a drone shot image at a given time step, segment the image and find global positions of all objects in
        image. Calculates velocity drone needs to fly to reach object. 
        '''
        # find center points of segmented image
        centers, _ = self.segment_image(seg)
        count = 0
        for x, y in centers:
            count += int(self.check_color(rgb[int(x), int(y)]))
        return count >= self.num_whales
    
    def pixel_to_world_velocity(self, pixel_coord, img_dims):
        '''
        Drone is at center of image, get velocity to target pixel coordinate
        '''
        m, n = img_dims[0], img_dims[1]
        x, y = pixel_coord
        dx, dy = -x + m/2, y - n/2
        incr_constant = np.sqrt(WHALE_TRACK_SPEED ** 2 / (dx ** 2 + dy ** 2))
        # self.write_debug_file([x, y, dx, dy])
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
            return [0, 0, 0, 0], float('inf')
        else:
            target_point = [avg_x / count, avg_y / count]
            dist_to_target = np.linalg.norm(np.array([avg_x / count, avg_y / count]) - np.array([seg.shape[0] // 2, seg.shape[1] // 2]))
            return self.pixel_to_world_velocity(target_point, seg.shape), dist_to_target   

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
    def check_centered(self, target_center, img_shape):
        return (target_center[0] - img_shape[0] // 2)**2 + (target_center[1] - img_shape[1] // 2)**2 < 10

    # track whale and return velocity to reach whale (based off order of whales along the x-axis)
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
            if self.check_centered(target_center, seg.shape):
                return [0, 0, 0, 0], True
            return self.pixel_to_world_velocity(target_center, seg.shape), False
        except:
            print("target center error")
            return [0, 0, 0, 0], False

    def track_whale_prev_center(self, seg, rgb):
        whale_centers, _ = self.get_whale_center_list(seg, rgb)
        # get point closest to prev target point
        min_dist = float('inf')
        target_center = whale_centers[0]
        for center in whale_centers:
            dist = np.linalg.norm(np.array(center) - np.array(self.prev_target_pixel_pos))
            if dist < min_dist:
                min_dist = dist
                target_center = center
        self.prev_target_pixel_pos = target_center
        return self.pixel_to_world_velocity(target_center, seg.shape) 

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
        if self.in_command is not None:
            command_type, text = self.in_command.split('|')
            if command_type == "velocity":
                x, y = text.split(',')
                return [float(x), float(y), 0, 0]
            
            elif command_type == "fly-to":
                x, y = text.split(',')
                return self.calc_velocity_to_point([float(x), float(y)])
        
            elif command_type == "num_whales":
                self.num_whales = int(text)
                return [0, 0, 0, 0]
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
    def __init__(self, drone_id, env, sim_dir, num_drones, lead_drone, init_position, formation_type, goal_assignment):
        super().__init__(drone_id=drone_id, env=env, num_drones=num_drones, sim_dir=sim_dir, lead_drone=lead_drone)
        self.search_state = 0 # 0: flying (1, 1), 1: flying -x direction, 2: flying x direction, 3: flying vertically, 4: turning
        self.start_vertical_timestep = 0 # timestep when drone starts moving vertically
        self.start_turning_timestep = 0
        self.target_y = 0
        self.init_velocity = [(9.0 - init_position[0]) / 24, (9 - init_position[1]) / 24, 0, 0]
        self.search_target_points = [] # (dx, dy) for the positions that each tagging drone should be at relative to search drone before tracking commences
        self.formation_type = formation_type
        self.prev_whale_count = 0        
        self.whale_count_observation_streak = 0 # number of consecutive time steps where whale count is the same
        self.gnn_model = NonLinearModel(attr_dim=16,max_edges=5,L=1) # only for goal assignment with GNN
        if goal_assignment == "gnn":
            self.gnn_model.load_state_dict(torch.load("pybullet_env/simulator/gnn/model_10agents_env4m_comvar_maxedges5_3conv_modelv2.pt"))
            self.gnn_model.eval()
            
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
        
    def check_whales(self, seg, rgb):
        '''
        Given a drone shot image at a given time step, segment the image and find global positions of all objects in
        image. Scout drone tracks whales until it confirms exactly how many whales there are. 
        '''
        # find center points of segmented image
        centers, _ = self.get_whale_center_list(seg, rgb)
        count = len(centers)
        if self.whale_count_observation_streak > WHALE_CHECK_THRESHOLD:
            self.num_whales = count
            # set all other drones num_whales to count
            self.send_command_all_drones("num_whales", str(count))
            return True
        elif count != 0 and count == self.prev_whale_count: 
            self.whale_count_observation_streak += 1
            return False
        else:
            self.prev_whale_count = count
            self.whale_count_observation_streak = 0
            return False

    def calc_search_target_points(self):
        if self.formation_type == "line":
            search_width = min(self.num_drones - 2, 2.5)
            dx = -search_width / 2
            dy = 0
            for _ in range(self.num_drones - 1):
                self.search_target_points.append((dx, dy))
                dx += search_width / (self.num_drones - 2)
        elif self.formation_type == "polygon":
            search_radius = 0.75
            variance = 0.25 
            for i in range(self.num_drones - 1):
                variance_dist = np.random.uniform(0, variance)
                angle = 2 * np.pi * i / (self.num_drones - 1)
                dx = (search_radius + variance_dist) * np.cos(angle)
                dy = (search_radius + variance_dist) * np.sin(angle)
                self.search_target_points.append((dx, dy))

    def all_drones_in_position(self, dist_to_target):
        search_drone_pos = self.get_drone_state()[:2]
        if np.linalg.norm(dist_to_target) > 2:
            return False
        for drone in self.other_drones:
            if drone != self.drone_id:
                drone_pos = self.other_drones[drone].get_drone_state()[:2]
                real_dx = abs(drone_pos[0] - search_drone_pos[0])
                real_dy = abs(drone_pos[1] - search_drone_pos[1])
                target_dx = abs(self.search_target_points[int(drone) - 1][0])
                target_dy = abs(self.search_target_points[int(drone) - 1][1])
                if real_dx > target_dx + 0.1 or real_dy > target_dy + 0.1:
                    return False
                if self.other_drones[drone].mode != "whales":
                    return False
        return True

    def all_drones_centered(self):
        for drone in self.other_drones:
            if drone != self.drone_id:
                if self.other_drones[drone].mode not in ["centered", "landing", "complete"]:
                    return False
        return True

    def all_drones_landed(self):
        for drone in self.other_drones:
            if drone != self.drone_id:
                if self.other_drones[drone].mode != "complete":
                    return False
        return True        

    def get_drone_to_whale_dists(self):
        '''
        Performs ICP to get consensus on whale positions between drones, and then calculates bounding boxes/distances
        between each whale and drone to then be used for goal assignment
        '''
        all_whale_boxes = []
        net_corr = [i for i in range(self.num_drones - 1)] # mapping from drone 1 positions to drone x positions aggregatively
        for d in range(1, self.num_drones):
            rgb1, _, seg1 = self.env._getDroneImages(d)

            if d + 1 == self.num_drones:
                rgb2, _, seg2 = self.env._getDroneImages(1)
            else:
                rgb2, _, seg2 = self.env._getDroneImages(d + 1)

            # get centers of whales
            whale_centers1, whale_boxes1 = self.get_whale_center_list(seg1, rgb1)
            whale_centers2, whale_boxes2 = self.get_whale_center_list(seg2, rgb2)  

            if len(whale_centers1) != len(whale_centers2):
                print(f"Different number of whales detected at index {d}, centers 1: {whale_centers1}, centers 2: {whale_centers2}")
                return False, None, None
            
            _, corr = icp(np.array(whale_boxes1), np.array(whale_boxes2))
            # calculate net correspondence
            new_correspondence = [0 for _ in range(self.num_drones - 1)]
            for i in range(len(corr)):
                new_correspondence[i] = net_corr[corr[i]]
            net_corr = new_correspondence.copy()

            # append all whale boxes to list with boxes repermuted according to 1st image label number
            all_whale_boxes.append([whale_boxes1[p] for p in net_corr])
            
        # check the mapping is the identity at the end
        if list(net_corr) != [i for i in range(self.num_drones - 1)]:
            print("net correlation was not the identity mapping: ", net_corr)
            return False, None, None
        
        # nearest neighbor for whichever point is closest to a given whale
        drone_to_whale_dists = np.zeros((self.num_drones - 1, len(all_whale_boxes)))  
        center_pixel = np.array([rgb1.shape[0] // 2, rgb1.shape[1] // 2])
        for i in range(self.num_drones - 1):
            for j in range(len(all_whale_boxes[i])):
                whale_center = np.mean(all_whale_boxes[i][j], axis=0)
                drone_to_whale_dists[i][j] = norm(whale_center - center_pixel)
        return drone_to_whale_dists, all_whale_boxes

    def gnn_analysis(self):
        '''
        Perform GNN analysis on drone images to check if drones agree on whale positions,
        then outputs velocity commands for each drone to target specific whale positions
        Assumes decentralized communication between drones   
        '''
        print("PERFORMING GNN ANALYSIS ON WHALE EDGES")
        drone_to_whale_dists, all_whale_boxes = self.get_drone_to_whale_dists()
        
        # convert graph to dgl format
        num_agents, num_targets = drone_to_whale_dists.shape
        src_ids = np.repeat(np.arange(num_agents), num_targets)
        target_ids = np.tile(np.arange(num_targets), num_agents)
        graph = dgl.heterograph({("agent", "observes", "target"): (src_ids, target_ids)})
        edge_distances = torch.tensor(drone_to_whale_dists.flatten(), dtype=torch.float32)
        graph.edges["observes"].data["distance"] = edge_distances
        with torch.no_grad():
            pred, edges = self.gnn_model(graph)
            assert False

    def icp_analysis(self):
        drone_to_whale_dists, all_whale_boxes = self.get_drone_to_whale_dists() 
        _, drone_col_assign = linear_sum_assignment(drone_to_whale_dists)
        velocity_vecs = []
        target_pixels = []
        for i in range(self.num_drones - 1):
            target_pixel = np.mean(all_whale_boxes[i][drone_col_assign[i]], axis=0)
            target_pixel = [target_pixel[1], target_pixel[0]]
            target_pixels.append(target_pixel)
            velocity_vecs.append(self.pixel_to_world_velocity(target_pixel, rgb1.shape)) # for some reason target pixel index values need to be swapped

        # draw target pixels to debug image
        for d in range(1, self.num_drones):
            rgb, _, _ = self.env._getDroneImages(d)
            self.other_drones[str(d)].prev_target_pixel_pos = target_pixels[d - 1]
            cv2.circle(rgb, (int(target_pixels[d - 1][1]), int(target_pixels[d - 1][0])), 2, (0, 0, 0), -1)
            cv2.imwrite(self.sim_dir + f"/center_pics{d}/icp_targets/target_pixels_{self.timestep}.png", rgb) 
        return True, velocity_vecs, target_pixels

    # send command to specific drone
    def send_command_drone(self, command, text, target_drone):
        self.other_drones[target_drone].in_command = f"{command}|{text}"

    # generalized send_command function to all drones
    def send_command_all_drones(self, command, text):
        for drone in self.other_drones:
            if drone != self.drone_id:
                self.other_drones[drone].in_command = f"{command}|{text}"

    # for lead drone to send fly commands to all drones during whale tracking phase
    def whale_stage_send_command(self, cur_pos):
        x, y = cur_pos[0], cur_pos[1]
        for i in range(1, self.num_drones):
            dx, dy = self.search_target_points[i - 1]
            self.send_command_drone("fly-to", f"{x + dx},{y + dy}", str(i))
    