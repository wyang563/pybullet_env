import pybullet as p
import cv2
import numpy as np
from simulator.simulator_utils import *
import csv
from pybullet_env.icp.icp import rot_icp 
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
import dgl
import torch
import networkx as nx
from .gnn.models.modelv2 import NonLinearModel
from .gnn.create_dataset import random_init, is_connected, create_graph
from icp.icp_test import plot_points
import json

# IMPORTANT CONSTANTS FOR RECON SEARCH
SEARCH_SPEED = 0.25
DEFAULT_SPEED = 0.12
WHALE_TRACK_SPEED = 0.10
WHALE_CHECK_THRESHOLD = 50 # Number of iterations whale count is the same before switching to whales mode
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
        self.prev_target_pixel_pos = None # for tracking stage (when using ICP), previous assigned drone target pixel position (global pixel coordinate of whale being targetted)
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
    
    # coordinate conversion/rotation functions
    def rel_to_global(self, vec):
        cur_yaw = self.env._getDroneStateVector(int(self.drone_id))[9]
        rot_matrix = np.array([[np.cos(-cur_yaw), -np.sin(-cur_yaw)], 
                               [np.sin(-cur_yaw), np.cos(-cur_yaw)]])
        global_vec = np.dot(rot_matrix, np.array(vec))
        return global_vec
    
    def global_to_rel(self, vec):
        cur_yaw = self.env._getDroneStateVector(int(self.drone_id))[9]
        rot_matrix = np.array([[np.cos(cur_yaw), -np.sin(cur_yaw)], 
                               [np.sin(cur_yaw), np.cos(cur_yaw)]])
        rel_vec = np.dot(rot_matrix, np.array(vec))
        return rel_vec
    
    def pixel_to_world_rel(self, pixel_coord, img_dims):
        # this assumes drone, or center of the image is (0, 0)
        m, n = img_dims[0], img_dims[1]
        x, y = pixel_coord
        return [y - n/2, -x + m/2]

    def world_rel_to_pixel(self, world_coord, img_shape):
        # also assumes drone is at origin of image
        m, n = img_shape[0], img_shape[1]
        return [round(-world_coord[1] + m/2), round(world_coord[0] + n/2)]
    
    def pixel_to_origin(self, pixel_coord, img_dims):
        # assumes bottom left corner is origin, cv2 img coords with (x = horizontal, y = vertical)
        m = img_dims[0]
        x, y = pixel_coord
        return [x, m - y]
    
    def origin_to_pixel(self, origin_coord, img_dims):
        # assumes bottom left corner is origin, cv2 img coords with (x = horizontal, y = vertical)
        m = img_dims[0]
        x, y = origin_coord
        return [x, m - y]
    
    def rotate_pixel(self, pixel_coord, img_dims, to_global=True):
        yaw = self.env._getDroneStateVector(int(self.drone_id))[9]
        if to_global:
            yaw = -yaw
        world_rel_coord = self.pixel_to_world_rel(pixel_coord, img_dims)
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                               [np.sin(yaw), np.cos(yaw)]])
        new_world_rel_coord = np.dot(rot_matrix, np.array(world_rel_coord))
        return self.world_rel_to_pixel(new_world_rel_coord, img_dims)

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
        point_cloud = []

        for label in range(1, num_labels):  # Skip label 0 (background)
            # Find pixels belonging to the current label
            object_pixels = np.argwhere(labels_im == label)

            # Calculate the center point
            center_x, center_y = object_pixels.mean(axis=0)
            centers.append((center_x, center_y))

            rect = cv2.minAreaRect(object_pixels[:, ::-1])  
            box = cv2.boxPoints(rect) 
            boxes.append(box.tolist())

            # create a point cloud of within a given label
            curr_point_cloud = []
            for _ in range(20):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, 15)
                px = int(center_y + r * np.cos(angle))
                py = int(center_x + r * np.sin(angle))
                curr_point_cloud.append([px, py])
            point_cloud.append(curr_point_cloud.copy())

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
                cv2.drawContours(colored_img, [np.int0(box)], 0, (0, 255, 0), 2)  # Draw the rectangle in green

            if self.timestep % 400 == 0:
                cv2.imwrite(self.sim_dir + f"/center_pics{self.drone_id}/segments_drone_{self.timestep}.png", colored_img) 
        return centers, boxes, point_cloud

    def check_whales(self, seg, rgb):
        '''
        THIS IMPLEMENTATION IS ONLY FOR TRACKING DRONES - SCOUTING DRONE IMPLEMENTATION BELOW
        Given a drone shot image at a given time step, segment the image and find global positions of all objects in
        image. Calculates velocity drone needs to fly to reach object. 
        '''
        # find center points of segmented image
        centers, _, _ = self.segment_image(seg)
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
        # prevent divide by zero errors
        if dx ** 2 + dy ** 2 < 1e-7:
            return [0, 0, 0, 0]
        incr_constant = np.sqrt(WHALE_TRACK_SPEED ** 2 / (dx ** 2 + dy ** 2))
        # rotate velocity according to drone's current orientation
        dx, dy = self.rel_to_global([dx, dy])
        return [dy * incr_constant, dx * incr_constant, 0, 0]

    def get_whale_center_list(self, seg, rgb, use_cloud=False):
        '''
        returns pixel coordinate list of whale centers along with their corresponding bounding boxes
        '''
        centers, boxes, point_cloud = self.segment_image(seg)
        whale_centers = []
        whale_boxes = []
        whale_point_clouds = []
        for i, (x, y) in enumerate(centers):
            if self.check_color(rgb[int(x), int(y)]):
                whale_centers.append((x, y))
                whale_boxes.append(boxes[i])
                whale_point_clouds.append(point_cloud[i])
        if use_cloud:
            return whale_centers, whale_point_clouds 
        return whale_centers, whale_boxes

    def get_whales_center(self, seg, rgb):
        '''
        get whale center and calculate velocity to rough area where whale center is
        '''
        centers, _, _ = self.segment_image(seg)
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
        return (target_center[0] - img_shape[0] // 2) ** 2 + (target_center[1] - img_shape[1] // 2) ** 2 < 10

    # track whale and return velocity to reach whale (based off order of whales along the x-axis)
    def track_whale(self, seg, rgb):
        centers, _, _ = self.segment_image(seg)
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
            # convert center to global coordination position
            dist = np.linalg.norm(center - np.array(self.rotate_pixel(self.prev_target_pixel_pos, rgb.shape, to_global=False)))
            if dist < min_dist:
                min_dist = dist
                target_center = center 
        self.prev_target_pixel_pos = self.rotate_pixel(target_center, rgb.shape, to_global=True)
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
    def __init__(self, drone_id, env, sim_dir, num_drones, lead_drone, init_position, formation_type, goal_assignment, gnn_model_path, search_type):
        super().__init__(drone_id=drone_id, env=env, num_drones=num_drones, sim_dir=sim_dir, lead_drone=lead_drone)
        self.target_y = 0
        self.init_velocity = [(9.0 - init_position[0]) / 24, (9 - init_position[1]) / 24, 0, 0]
        self.search_target_points = [] # (dx, dy) for the positions that each tagging drone should be at relative to search drone before tracking commences
        self.formation_type = formation_type
        self.prev_whale_count = 0        
        self.whale_count_observation_streak = 0 # number of consecutive time steps where whale count is the same
        self.gnn_model = NonLinearModel(attr_dim=16,max_edges=5,L=5) # only for goal assignment with GNN
        if goal_assignment == "gnn":
            if torch.cuda.is_available():
                self.gnn_model.load_state_dict(torch.load(gnn_model_path))
            else:
                self.gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=torch.device('cpu')))
            self.gnn_model.eval()
        if search_type == "spiral":
            # spiral search parameters
            start_radius = 3
            growth_rate = 3
            num_points = 50 
            num_turns = 4
            self.spiral_points = self.generate_spiral(start_radius, growth_rate, num_points, num_turns)
            self.target_point = 0 # index of target point in spiral_points drone should go to
            self.search_state = None
        else:
            self.search_state = 0 # 0: flying (1, 1), 1: flying -x direction, 2: flying x direction, 3: flying vertically, 4: turning
            self.start_vertical_timestep = 0 # timestep when drone starts moving vertically
            self.start_turning_timestep = 0
            
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
    
    def generate_spiral(self, start_radius, growth_rate, num_points, num_turns):
        theta = np.linspace(0, 2 * np.pi * num_turns, num_points)
        radius = start_radius + growth_rate * theta  # Radius increases with theta
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        return x, y

    def search_spiral(self):
        x, y = self.get_drone_state()[:2]
        if np.linalg.norm(np.array([x, y]) - np.array([self.spiral_points[0][self.target_point], self.spiral_points[1][self.target_point]])) < 0.05:
            self.target_point += 1
            if self.target_point == len(self.spiral_points):
                self.target_point = 0
        return self.calc_velocity_to_point([self.spiral_points[0][self.target_point], self.spiral_points[1][self.target_point]])

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
            search_radius = 0.3 
            variance = 0.1 
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
        origin_center_all_whale_boxes = []
        origin_center_all_whale_centers = []
        correlations = []

        # setup debug directories in icp plots
        run_num = len(os.listdir("pybullet_env/icp/sim_data"))
        os.makedirs(f"pybullet_env/icp/sim_data/run_{run_num}")

        for d in range(1, self.num_drones):
            rgb1, _, seg1 = self.env._getDroneImages(d)

            # save whale image
            cv2.imwrite(f"pybullet_env/icp/sim_data/run_{run_num}/drone_{d}_icp_whale_image.png", rgb1)
            cv2.imwrite(self.sim_dir + f"/icp_plots/drone_{d}_icp_whale_image.png", rgb1)
            if d + 1 == self.num_drones:
                rgb2, _, seg2 = self.env._getDroneImages(1)
            else:
                rgb2, _, seg2 = self.env._getDroneImages(d + 1)

            # get centers of whales
            whale_centers1, whale_boxes1 = self.get_whale_center_list(seg1, rgb1, use_cloud=False)
            whale_centers2, whale_boxes2 = self.get_whale_center_list(seg2, rgb2, use_cloud=False)  

            # convert coordinates to global coordinates with origin in bottom right 
            origin_whale_boxes1 = [[self.pixel_to_origin([x, y], rgb1.shape) for x, y in box] for box in whale_boxes1]
            origin_whale_boxes2 = [[self.pixel_to_origin([x, y], rgb2.shape) for x, y in box] for box in whale_boxes2]
            origin_whale_centers1 = [self.pixel_to_origin([x, y], rgb1.shape) for x, y in whale_centers1]
            origin_whale_centers2 = [self.pixel_to_origin([x, y], rgb2.shape) for x, y in whale_centers2]

            if len(whale_centers1) != len(whale_centers2):
                print(f"Different number of whales detected at index {d}, centers 1: {whale_centers1}, centers 2: {whale_centers2}")
                return False, None, None, None

            # _, corr, _ = rot_icp(np.array(origin_whale_boxes2), np.array(origin_whale_boxes1), N=50)
            _, corr, _ = rot_icp(np.array(origin_whale_boxes2), np.array(origin_whale_boxes1), N=50, use_point=False)

            # debug plot correlations
            plot_points(d, np.array(origin_whale_boxes1), np.array(origin_whale_boxes2), corr, title1=f"Set {d}", title2=f"Set {1 if d + 1 == self.num_drones else d + 1}", sim_dir=f"pybullet_env/icp/sim_data/run_{run_num}")

            correlations.append(corr)

            # append all whale boxes to list with boxes repermuted according to 1st image label number
            origin_center_all_whale_boxes.append(origin_whale_boxes1) 
            origin_center_all_whale_centers.append(origin_whale_centers1)
            all_whale_boxes.append(whale_boxes1)

        # calculate net correlations
        net_corrs = [] # net_corrs[i] is the mapping from point cloud i to point cloud 0
        for i in range(len(correlations)):
            composite = np.arange(len(correlations[i]))
            for j in range(i, -1, -1):
                new_composite = np.zeros(len(composite), dtype=int)
                for k in range(len(composite)):
                    new_composite[correlations[j][k]] = composite[k]
                composite = new_composite.copy()
            net_corrs.append(composite)
        
        front = net_corrs.pop()
        net_corrs.insert(0, front)
        print(net_corrs)
         
        with open(f"pybullet_env/icp/sim_data/run_{run_num}/points.json", "w") as f:      
            json.dump(origin_center_all_whale_boxes, f, indent=4)
        
        with open(f"pybullet_env/icp/sim_data/run_{run_num}/centers.json", "w") as f:
            json.dump(origin_center_all_whale_centers, f, indent=4)

        with open(f"pybullet_env/icp/sim_data/run_{run_num}/correlations.json", "w") as f:
            json.dump([c.tolist() for c in net_corrs], f, indent=4)

        # check the mapping is the identity at the end
        if net_corrs[0].tolist() != [i for i in range(self.num_drones - 1)]:
            assert False, f"final net correlation was not the identity mapping: {net_corrs[0]}"
        
        # reshuffle all_whale_boxes
        for i in range(len(all_whale_boxes)):
            corr = net_corrs[i]
            all_whale_boxes[i] = [all_whale_boxes[i][corr[j]] for j in range(len(all_whale_boxes[i]))]

        print("ICP whale consensus succesful!")
        # debugging, plot correlations for all_whale_boxes        
        # for i in range(self.num_drones - 1):
        #     A_points = all_whale_boxes[i % (self.num_drones - 1)] 
        #     B_points = all_whale_boxes[(i + 1) % (self.num_drones - 1)] 
        #     plot_points(i, np.array(A_points), np.array(B_points), [i for i in range(self.num_drones - 1)], sim_dir=self.sim_dir)

        # nearest neighbor for whichever point is closest to a given whale
        drone_to_whale_dists = np.zeros((self.num_drones - 1, len(all_whale_boxes)))  
        center_pixel = np.array([rgb1.shape[0] // 2, rgb1.shape[1] // 2])
        for i in range(self.num_drones - 1):
            for j in range(len(all_whale_boxes[i])):
                whale_center = np.mean(all_whale_boxes[i][j], axis=0)
                drone_to_whale_dists[i][j] = norm(whale_center - center_pixel)
        return True, drone_to_whale_dists, all_whale_boxes, net_corrs, rgb1.shape

    # GNN Analysis Functions    
    def visualize_dgl_graph(self, g, out_file="pybullet_env/simulator/gnn/graph.png"):
        # Convert the DGL graph to a NetworkX graph
        nx_graph = g.to_networkx()
        
        # Create a layout for our nodes 
        pos = nx.spring_layout(nx_graph)
        
        # Draw the graph with node labels
        plt.figure(figsize=(8, 6))
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            node_size=500,
            font_size=10
        )
    
        # Save the plot to a PNG file
        plt.savefig(out_file)
        plt.close()

    def compute_comm_edges(self):
        u_comm = []
        v_comm = []
        for i in range(self.num_drones - 1):
            for j in range(self.num_drones - 1):
                if j in [(i + k) % self.num_drones for k in range(self.num_drones - 1)]:
                    u_comm.append(i)
                    v_comm.append(j)
        return torch.tensor(u_comm, dtype=torch.int32), torch.tensor(v_comm, dtype=torch.int32)

    def construct_graph(self, cost_matrix, attr_dim=16):
        nAgents = self.num_drones - 1
        u = torch.tensor([0 for _ in range(nAgents)])
        v = torch.tensor([i for i in range(nAgents)])
        for i in range(1,nAgents):
            u = torch.cat((u,torch.tensor([i for _ in range(nAgents)])))
            v = torch.cat((v,torch.tensor([k for k in range(nAgents)])))
        u_com,v_com = self.compute_comm_edges() 
        graph_data = {('agent', 'assigns', 'goal'): (u, v), ('goal', 'assigns', 'agent'): (v, u),('agent', 'communicates', 'agent'): (u_com, v_com)}
        graph = dgl.heterograph(graph_data,idtype=torch.int32)
        
        graph.nodes['agent'].data['hv']= torch.zeros(nAgents,attr_dim)
        graph.nodes['goal'].data['hv']=torch.zeros(nAgents,attr_dim)
        
        num_edges = graph.num_edges(('agent','assigns','goal'))
        dist_edges = torch.zeros(num_edges,1)
        for i in range(num_edges):
            agent_id = graph.edges(etype=('agent','assigns','goal'))[0][i]
            goal_id = graph.edges(etype=('agent','assigns','goal'))[1][i]
            dist_edges[i,0] = cost_matrix[agent_id,goal_id]
        
        graph.edges[('agent','assigns','goal')].data['he'] = dist_edges
        graph.edges[('goal','assigns','agent')].data['he'] = dist_edges 
        return graph

    def get_assigned_goals(self, h, edges):
        assignments = []
        for i in range(len(h)):
            agent = edges[0][i].item()
            goal = edges[1][i].item()
            likelihood = h[i].item()
            assignments.append((agent, goal, likelihood))

        # Find the highest likelihood assignment for each agent
        best_assignments = {}
        for assignment in assignments:
            agent, goal, likelihood = assignment
            if agent not in best_assignments or likelihood > best_assignments[agent][1]:
                best_assignments[agent] = (goal, likelihood)
        return best_assignments

    def gnn_analysis(self):
        '''
        Perform GNN analysis on drone images to check if drones agree on whale positions,
        then outputs velocity commands for each drone to target specific whale positions
        Assumes decentralized communication between drones   
        '''
        print("PERFORMING GNN ANALYSIS ON WHALE CENTERS")

        # create dgl graph object (as per how it is done in create_dataset.py)
        success, cost_matrix, all_whale_boxes, net_corrs, rgb_shape = self.get_drone_to_whale_dists()
        if not success:
            return False, None, None 
        nAgents, nGoals = self.num_drones - 1, self.num_drones - 1 
        attr_dim = 16

        # normalize cost matrix
        cost_matrix = cost_matrix / np.max(cost_matrix) 
        graph = self.construct_graph(cost_matrix, attr_dim)
        assignments = []
        while len(assignments) != nAgents:
            with torch.no_grad():
                # self.visualize_dgl_graph(graph)
                h, edges = self.gnn_model.forward(graph)
                assignments = self.get_assigned_goals(h, edges)

        print("GNN assignments: ", assignments)
        velocity_vecs = []
        target_pixels = []
        for i in range(self.num_drones - 1):
            assigned_goal = assignments[i][0]
            target_pixel = np.mean(all_whale_boxes[i][assigned_goal], axis=0)
            # target_pixel = [target_pixel[1], target_pixel[0]]
            target_pixels.append(target_pixel)
            velocity_vecs.append(self.pixel_to_world_velocity(target_pixel, rgb_shape)) # for some reason target pixel index values need to be swapped

        # draw target pixels to debug image
        _, axes = plt.subplots(1, self.num_drones - 1, figsize=(5 * (self.num_drones - 1), 5))

        if self.num_drones - 1 == 1:
            # If there's only one drone, put axes in a list for consistency when indexing
            axes = [axes]

        for idx, d in enumerate(range(1, self.num_drones)):
            # Get the RGB image for drone d
            rgb, _, _ = self.env._getDroneImages(d)

            self.other_drones[str(d)].prev_target_pixel_pos = self.rotate_pixel(target_pixels[idx], rgb.shape, to_global=True)

            # Draw correspondences
            for i, box in enumerate(all_whale_boxes[idx]):
                center = np.mean(box, axis=0)
                cv2.putText(rgb, str(i), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Draw the target pixel (circle) on the drone's image
            cv2.circle(rgb, (int(target_pixels[idx][0]), int(target_pixels[idx][1])), 2, (0, 0, 0), -1)
            
            # Convert BGR to RGB for plotting with matplotlib
            rgb_plot = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(rgb_plot)
            axes[idx].set_title(f"Drone {d} Target Pixel")
            axes[idx].axis('off')

        plt.tight_layout()
        # Save one combined image with subplots
        plt.savefig(f"{self.sim_dir}/icp_plots/all_target_pixels_{self.timestep}.png")
        plt.close()
        return True, velocity_vecs, target_pixels

    # pure ICP analysis function
    def icp_analysis(self):
        success, drone_to_whale_dists, all_whale_boxes, net_corrs, rgb_shape = self.get_drone_to_whale_dists() 
        if not success:
            return False, None, None
        _, drone_col_assign = linear_sum_assignment(drone_to_whale_dists)
        velocity_vecs = []
        target_pixels = []
        
        # calculate target pixels
        for i in range(self.num_drones - 1):
            target_pixel = np.mean(all_whale_boxes[i][drone_col_assign[i]], axis=0)
            # target_pixel = [target_pixel[1], target_pixel[0]]
            target_pixels.append(target_pixel)
            velocity_vecs.append(self.pixel_to_world_velocity(target_pixel, rgb_shape)) # for some reason target pixel index values need to be swapped

        # Create a single figure with subplots for each drone
        _, axes = plt.subplots(1, self.num_drones - 1, figsize=(5 * (self.num_drones - 1), 5))

        if self.num_drones - 1 == 1:
            # If there's only one drone, put axes in a list for consistency when indexing
            axes = [axes]

        for idx, d in enumerate(range(1, self.num_drones)):
            # Get the RGB image for drone d
            rgb, _, _ = self.env._getDroneImages(d)

            self.other_drones[str(d)].prev_target_pixel_pos = self.rotate_pixel(target_pixels[idx], rgb.shape, to_global=True)

            # Draw correspondences
            for i, box in enumerate(all_whale_boxes[idx]):
                center = np.mean(box, axis=0)
                cv2.putText(rgb, str(i), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Draw the target pixel (circle) on the drone's image
            cv2.circle(rgb, (int(target_pixels[idx][0]), int(target_pixels[idx][1])), 2, (0, 0, 0), -1)
            
            # Convert BGR to RGB for plotting with matplotlib
            rgb_plot = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(rgb_plot)
            axes[idx].set_title(f"Drone {d} Target Pixel")
            axes[idx].axis('off')

        plt.tight_layout()
        # Save one combined image with subplots
        plt.savefig(f"{self.sim_dir}/icp_plots/all_target_pixels_{self.timestep}.png")
        plt.close()
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
    