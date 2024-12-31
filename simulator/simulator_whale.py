import os
import sys
import time
from datetime import datetime
import csv
import random
import numpy as np
import pandas as pd
import pybullet as p
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torchvision
from tqdm import trange

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from simulator.simulator_utils import *
from simulator.whale_model import WhaleDroneModel, WhaleDroneLeadModel

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_SAMPLING_FREQ_HQ = 3
DEFAULT_COLAB = False
DEFAULT_PARAMS_PATH = None
DEFAULT_CHECKPOINT_PATH = None
POINT_CLOUD_REGISTRATION = False

SCOUT_H = 6.0
H = 4.0
vanish_mode = False

def run_pybullet_only_hike(
        loc_color_tuple,
        output_folder=None,
        normalize_path=None,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=None,
        colab=DEFAULT_COLAB,
        record_hz = DEFAULT_SAMPLING_FREQ_HQ
):
    ordered_objs, ordered_locs = loc_color_tuple
    print(f"ordered_objs: {ordered_objs}")
    print(f"ordered_locs: {ordered_locs}")

    #! Trajectory-specific parameters
    #* Env Params
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, num_drones)

    Theta = random.random() * 2 * np.pi
    Theta = 0
    Theta0 = 0 # don't rotate the drone
    Theta_offset = 0 #random.choice([0.175 * np.pi, -0.175 * np.pi])
    
    # ! Initialize drone locations + starting cube object
    y_offset = 0.1
    x_offset = random.uniform(0, 1)
    rel_drone_locs = [(x_offset, y_offset), (x_offset - 1, y_offset), (x_offset + 1, y_offset)]
    rel_drone_locs = rel_drone_locs[:num_drones]
    ordered_objs.append("cube")
    ordered_locs.append((x_offset, y_offset))

    #* Save starting env params
    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        print(ordered_objs)
        f.write(str("".join(ordered_objs)))
        
    #* Object setup
    obj_loc_global = [convert_to_global(obj_loc_rel, Theta) for obj_loc_rel in ordered_locs]
    TARGET_LOCATIONS = obj_loc_global
    print(f"TARGET_LOCATIONS: {TARGET_LOCATIONS}")
    INIT_XYZS = []
    for i, rel_pos in enumerate(rel_drone_locs):
        if i == 0:
            height = SCOUT_H
        else:
            height = H
        INIT_XYZS.append([*convert_to_global(rel_pos, Theta), height])
    INIT_XYZS = np.array(INIT_XYZS)
    
    INIT_RPYS = np.array([[0, 0, Theta0 + Theta_offset] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    NUM_WP = control_freq_hz * duration_sec

    #### Create the environment with or without video capture ##
    if vision:
        env = VisionAviary(drone_model=drone,
                           num_drones=num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=physics,
                           neighbourhood_radius=10,
                           freq=simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=gui,
                           record=record_video,
                           obstacles=obstacles
                           )
    else:
        env = CtrlAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         neighbourhood_radius=10,
                         freq=simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         custom_obj_location=None if vanish_mode else
                            {
                                "colors": ordered_objs,
                                "locations": obj_loc_global
                            }
                        )
    
    #### Initialize the model #############################
    drone_models = {}
    for i in range(num_drones):
        if i == 0:
            drone_models[str(i)] = WhaleDroneLeadModel(drone_id=str(i), env=env, sim_dir=sim_dir, lead_drone=True, init_position=rel_drone_locs[i]) 
        else:
            drone_models[str(i)] = WhaleDroneModel(drone_id=str(i), env=env, sim_dir=sim_dir, lead_drone=False) 

    for i in range(num_drones):
        drone_models[str(i)].set_other_drones(drone_models)

    env.IMG_RES = np.array([256, 144])

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz))
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / DEFAULT_SAMPLING_FREQ_HQ))
    print("CTRL EVERY N Steps: ", CTRL_EVERY_N_STEPS)
    print("REC EVERY N Steps: ", REC_EVERY_N_STEPS)
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    # STEPS = CTRL_EVERY_N_STEPS * NUM_WP
    # STEPS = int(100 * 90 * 240.0 * record_hz / 8.0) * 2
    # STEPS = int(100 * 90 * 90) * 2
    # 30 seconds * 240 steps/sec
    STEPS = int(100 * 30 * 240) * 2
    x_data = [[] for _ in range(num_drones)]
    y_data = [[] for _ in range(num_drones)]
    value = np.array([0, 0, 0, 0, 0, 0])
    value = value[None,:]

    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        # State of drone at a time step
        # np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :], self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])

        #### Step the simulation ###################################
        obs, _, _, _ = env.step(action)
        states = [obs[str(d)]["state"] for d in range(num_drones)]
        for d in range(num_drones):
            drone_models[str(d)].set_timestep()
        #### Compute control at the desired frequency ##############
        if i % REC_EVERY_N_STEPS == 0:
            out = [[0 for _ in range(4)] for _ in range(num_drones)]
            if drone_models[str(0)].mode == "search":
                # get lead drone image
                rgb, _, seg = env._getDroneImages(0)
                if i % (REC_EVERY_N_STEPS * 10) == 0:
                    env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=f'{sim_dir}/pics0_search',
                                    frame_num=int(i / CTRL_EVERY_N_STEPS),
                                    )

                pred = drone_models["0"].check_whales(seg, rgb)
                if pred:
                    print("LEAD DRONE DETECTED WHALES!!")
                    drone_models["0"].mode = "whales"
                else:
                    out[0] = drone_models["0"].search_step(i)
                
            elif drone_models[str(0)].mode == "whales":
                for d in range(num_drones):
                    rgb, _, seg = env._getDroneImages(d)
                    if i % (REC_EVERY_N_STEPS * 10) == 0:
                        env._exportImage(img_type=ImageType.RGB,
                                        img_input=rgb,
                                        path=f'{sim_dir}/pics{d}_track',
                                        frame_num=int(i / CTRL_EVERY_N_STEPS),
                                        )
                        

                    # lead drone tracks whale
                    if d == 0:
                        out[d] = drone_models[str(d)].get_whales_center(seg, rgb)
                        cur_pos = drone_models[str(d)].get_drone_state()[:2]
                        drone_models[str(d)].track_stage_send_command(cur_pos)

                    else:
                        # follower drone receives command from scout drone
                        out[d] = drone_models[str(d)].receive_command()

                # check all other drones are in view of drone
                if drone_models["0"].all_drones_in_position():
                    print("ALL DRONES IN POSITION: SWITCHING TO TRACKING MODE")
                    for d in range(num_drones):
                        drone_models[str(d)].mode = "tracking"

            # tracking mode
            elif drone_models[str(0)].mode == "tracking":
                for d in range(num_drones):
                    rgb, _, seg = env._getDroneImages(d)
                    if i % (REC_EVERY_N_STEPS * 10) == 0:
                        env._exportImage(img_type=ImageType.RGB,
                                        img_input=rgb,
                                        path=f'{sim_dir}/pics{d}_track',
                                        frame_num=int(i / CTRL_EVERY_N_STEPS),
                                        )
                    if d == 0:
                        out[d] = drone_models[str(d)].get_whales_center(seg, rgb)
                    elif drone_models[str(d)].mode == "tracking":
                        out[d], centered = drone_models[str(d)].track_whale(seg, rgb)
                        if centered:
                            drone_models[str(d)].mode = "landing"
                            print(f"Drone {d} is centered on whale: switching to landing mode")
                    # landing mode
                    else:
                        out[d], landed = drone_models[str(d)].land_drone()
                        if landed:
                            print(f"Drone {d} has landed")
                            drone_models[str(d)].mode = "complete"
                            out[d] = [0, 0, 0, 0]
            else:
                raise Exception("Invalid mode")

        if i % CTRL_EVERY_N_STEPS == 0:
            for d in range(num_drones):
                action[str(d)], _, _ = ctrl[d].computeControl(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                            cur_pos=states[d][0:3],
                                            cur_quat=states[d][3:7],
                                            cur_vel=states[d][10:13],
                                            cur_ang_vel=states[d][13:16],
                                            target_pos=states[d][:3],  # same as the current position
                                            target_rpy=np.array([0, 0, 0]),  # keep current yaw
                                            target_vel=out[d][:3],
                                            target_rpy_rates=np.array([0, 0, 0])
                                            )
                x_data[d].append(states[d][0])
                y_data[d].append(states[d][1])
                    
            # apply external forces to ball objects to simulate whale movement with random force direction
            fx = random.uniform(12, 18)
            fx_sign = random.choice([-1, 1])
            fy = random.uniform(12, 18)
            fy_sign = random.choice([-1, 1])
            B_pos = p.getBasePositionAndOrientation(env.object_ids["B"])[0]
            G_pos = p.getBasePositionAndOrientation(env.object_ids["G"])[0]
            p.applyExternalForce(objectUniqueId=env.object_ids["B"], linkIndex=-1, forceObj=[fx_sign * fx, fy_sign * fy, 0], posObj=B_pos, flags=p.WORLD_FRAME)
            p.applyExternalForce(objectUniqueId=env.object_ids["G"], linkIndex=-1, forceObj=[fx_sign * fx, fy_sign * fy, 0], posObj=G_pos, flags=p.WORLD_FRAME)

            # plot path on grid
            if i % (CTRL_EVERY_N_STEPS * 100) == 0:
                for d in range(num_drones):
                    with open(sim_dir + f'/state{d}.csv', mode='a') as state_file:
                        state_writer = csv.writer(state_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        state_writer.writerow([i, *states[d]])
 
                    with open(sim_dir + f'/vel_cmd{d}.csv', mode='a') as vel_cmd_file:
                        vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        vel_cmd_writer.writerow([i, *out[d]])
                
                B_pos = p.getBasePositionAndOrientation(env.object_ids["B"])[0]
                G_pos = p.getBasePositionAndOrientation(env.object_ids["G"])[0]
                with open(sim_dir + '/target_pos.csv', mode='a') as f:
                    target_pos_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    target_pos_writer.writerow([i, *B_pos, *G_pos, fx, fy])
                    
            # plot path on grid TODO: update for all drones later
            
            if i % (CTRL_EVERY_N_STEPS * 500) == 0:
                # Create one figure
                plt.figure(figsize=(8, 6))
                
                for d in range(num_drones):
                    data = pd.read_csv(os.path.join(sim_dir, f"state{d}.csv"))
                    
                    x = data.iloc[:, 1]
                    y = data.iloc[:, 2]
                    
                    # Plot each drone's path on the same figure
                    plt.plot(x, y, label=f"Drone {d} Path")
                
                # plot target path
                for i in range(2):
                    data = pd.read_csv(os.path.join(sim_dir, f"target_pos.csv"))
                    x = data.iloc[:, 1 + 3 * i]
                    y = data.iloc[:, 2 + 3 * i]
                    plt.plot(x, y, label=f"Target {i} Path")
                
                # Plot the target locations
                plt.scatter(B_pos[0], B_pos[1], label="Target1", color='blue')
                plt.scatter(G_pos[0], G_pos[1], label="Target2", color='green')
                
                # Label, title, legend
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Drone Paths")
                plt.legend()
                
                # Save and close
                plt.savefig(os.path.join(sim_dir, "all_drones_paths.jpg"), dpi=300)
                plt.close()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

        # check if all drones have landed
        if drone_models["0"].all_drones_landed():
            break

    env.close()


