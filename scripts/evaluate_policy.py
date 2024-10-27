import os
import argparse
import math
import random
import numpy as np
import glob
from PIL import Image, ImageDraw, ImageFont
import joblib
import subprocess
from tqdm import tqdm
import tomli

from simulator.runna_hike import run_pybullet_only_hike


def get_tag_names(config):
    saved_model_folders_list = config['saved_model_folders_list']

    tag_names = ["_".join(folder.split('/')[-1].split('_')[1:]) for folder in saved_model_folders_list]
    tag_names = [name + "_multi" if config['multi_step_objects'] else name for name in tag_names]
    return tag_names

def get_output_folders(tag_names, config):
    return [f'{config["output_dir"]}/cl_real_{tag}' for tag in tag_names]

def expand_to_distributed_setup(config):
    """ 
    Takes a list of models to evaluate, how many times to evaluate each model, other settings, and
    returns the expanded lists for distributing the runs across multiple processes
    """
    concurrent_params_paths = []
    concurrent_checkpoint_paths = []
    output_folder_paths = []
    expanded_record_hzs = []
    expanded_variable_timesteps = []

    for base_folder, output_folder, record_hz, variable_timestep in zip(
            config['saved_model_folders_list'], 
            get_output_folders(get_tag_names(config), config),
            config['record_hzs'], 
            config['variable_timesteps']):

        val_folder = os.path.join(base_folder, 'val')
        hdf5_files = glob.glob(os.path.join(val_folder, '*.hdf5'))
        json_files = glob.glob(os.path.join(val_folder, '*.json'))

        if hdf5_files and json_files and not os.path.exists(os.path.join(output_folder, 'val')):
            for _ in range(config['runs_per_model']):
                concurrent_checkpoint_paths.append(hdf5_files[0])
                concurrent_params_paths.append(json_files[0]) 
                output_folder_paths.append(os.path.join(output_folder, 'val'))
                expanded_record_hzs.append(record_hz)
                expanded_variable_timesteps.append(variable_timestep)

        # Debug code for evaluating from checkpoints that are saved periodically every X epochs
        # # Recurrent checkpoints evaluation
        # recurrent_folder = os.path.join(base_folder, 'recurrent')
        # for hdf5_file in glob.glob(os.path.join(recurrent_folder, '*.hdf5')):
        #     epoch_num = int(re.findall(r'epoch-(\d+)', hdf5_file)[0])
            
        #     if os.path.exists(os.path.join(output_folder, f'recurrent{epoch_num}')):
        #         continue

        #     params_file = os.path.join(base_folder, 'recurrent', f'params{epoch_num}.json')
        #     if os.path.exists(params_file):
        #         for _ in range(RUNS_PER_MODEL):
        #             concurrent_checkpoint_paths.append(hdf5_file)
        #             concurrent_params_paths.append(params_file)
        #             output_folder_paths.append(os.path.join(output_folder, f'recurrent{epoch_num}'))
        #             expanded_record_hzs.append(record_hz)
        #             expanded_variable_timesteps.append(variable_timestep)

    return (concurrent_params_paths, concurrent_checkpoint_paths, 
            output_folder_paths, expanded_record_hzs, expanded_variable_timesteps)

def generate_trajectories(config):
    if config['multi_step_objects']:
        objects = config['objects']
        locations_rel = []
        for targets in objects:
            locations = []
            cur_point = (0, 0)
            cur_direction = 0
            for target in targets:
                cur_dist = random.uniform(*config['starting_distance_range']) - 0.2
                target_loc = (
                    cur_point[0] + (cur_dist + 0.2) * math.cos(cur_direction),
                    cur_point[1] + (cur_dist + 0.2) * math.sin(cur_direction)
                )
                cur_point = (
                    cur_point[0] + cur_dist * math.cos(cur_direction),
                    cur_point[1] + cur_dist * math.sin(cur_direction)
                )
                locations.append(target_loc)
                
                if target == 'R':
                    cur_direction += config['multi_step_angle_between']
                elif target == 'B':
                    cur_direction -= config['multi_step_angle_between']
                    
            locations_rel.append(locations)
    else:
        # TODO: Fix with config if need to use single-step eval
        objects = [['R'], ['B']] * (len(output_folder_paths) // 2)
        locations_rel = [[(random.uniform(*config['starting_distance_range']), 0)] for _ in range(len(output_folder_paths))]

    return objects, locations_rel

def process_videos(output_folders):
    for folder in output_folders:
        try:
            for eval_dir in os.listdir(folder):
                eval_path = os.path.join(folder, eval_dir)
                
                for run_dir in sorted(os.listdir(eval_path)):
                    run_path = os.path.join(eval_path, run_dir)
                    
                    if not os.path.isdir(run_path):
                        continue
                        
                    # Generate video if not exists
                    if "rand.mp4" not in os.listdir(run_path):
                        pics_path = os.path.join(run_path, "pics0")
                        labeled_path = os.path.join(run_path, "labeled_pics")
                        
                        os.makedirs(labeled_path, exist_ok=True)
                        
                        # Add labels to images
                        for i, img_name in enumerate(sorted(os.listdir(pics_path))):
                            img = Image.open(os.path.join(pics_path, img_name))
                            if i < 8:
                                draw = ImageDraw.Draw(img)
                                # TODO: Find alternative to hardcoded font
                                font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 20)
                                draw.text((img.width - 60, 10), "begin", fill="red", font=font)
                            img.save(os.path.join(labeled_path, img_name))
                            
                        # Generate video
                        os.system(f"ffmpeg -framerate 240 -pattern_type glob -i '{labeled_path}/0*.png' "
                                f"-c:v libx264 -pix_fmt yuv420p {run_path}/rand.mp4 > /dev/null 2>&1")
                        os.system(f"rm -rf {labeled_path}")
                
                # Combine videos
                video_paths = [os.path.join(eval_path, d, "rand.mp4") 
                             for d in sorted(os.listdir(eval_path))
                             if os.path.isdir(os.path.join(eval_path, d)) and
                             "rand.mp4" in os.listdir(os.path.join(eval_path, d))]
                
                with open("input.txt", "w") as f:
                    f.write("\n".join(f"file {path}" for path in video_paths))
                    
                subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt",
                              "-c", "copy", f"{eval_path}/combined_video.mp4"])
                              
        except Exception as e:
            print(f"Error processing {folder}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories')
    parser.add_argument('--config', type=str, default='configs/evaluate_policy.toml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        config = tomli.load(f)

    # Get model paths and settings
    paths = expand_to_distributed_setup(config)
    concurrent_params_paths, concurrent_checkpoint_paths, output_folder_paths, expanded_record_hzs, expanded_variable_timesteps = paths

    # Generate trajectories
    objects, locations_rel = generate_trajectories(config)

    # Run simulations in parallel
    total_list = [(obj, loc) for output_folder in output_folder_paths
                 for obj, loc in zip(objects, locations_rel)]

    joblib.Parallel(n_jobs=config['n_jobs'])(
        joblib.delayed(run_pybullet_only_hike)(
            d, 
            output_folder=output_folder_path,
            params_path=params_path,
            checkpoint_path=checkpoint_path,
            duration_sec=config['duration_sec'], # TODO: Remove since it's not being used
            record_hz=record_hz
        )
        for d, params_path, checkpoint_path, output_folder_path, record_hz, variable_timestep 
        in tqdm(zip(total_list, concurrent_params_paths, concurrent_checkpoint_paths,
                   output_folder_paths, expanded_record_hzs, expanded_variable_timesteps))
    )

    # Process and combine videos
    process_videos(get_output_folders(get_tag_names(config), config))
