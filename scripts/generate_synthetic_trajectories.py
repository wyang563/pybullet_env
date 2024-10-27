
import os
import random
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import json
import argparse
import tomli

from simulator.simulator_base import DEFAULT_NUM_DRONES
from simulator.simulator_train import TrainSimulator
from simulator.simulator_utils import setup_folders
from path_templates.trajectory_templates import get_init_conditions_func


def generate_one_training_trajectory(config, object_tags):
    """
    Requires config to have the following keys:
    - base_dir
    - samples
    - record_hz
    - task_tag
    """
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(config['base_dir'], sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)

    generate_init_conditions_func = get_init_conditions_func(config['task_tag'])
    init_conditions = generate_init_conditions_func(object_tags)
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'w') as f:
        json.dump(init_conditions, f)

    sim = TrainSimulator(sim_dir, init_conditions, config['record_hz'], config['task_tag'])
    
    sim.precompute_trajectory()
    sim.run_simulation_to_completion()

    sim.export_plots()
    sim.logger.save_as_csv(sim_name, sim.custom_timesteps if sim.custom_timesteps else None)  # Optional CSV save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories')
    parser.add_argument('--config', type=str, default='configs/generate.toml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        config = tomli.load(f)

    total_list = config['object_tags'] * (config['samples'] // len(config['object_tags']))
    random.shuffle(total_list)

    joblib.Parallel(n_jobs=config['n_jobs'])(joblib.delayed(generate_one_training_trajectory)(config, object_tags) for object_tags in tqdm(total_list))
