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
from simulator.simulator_whale import run_pybullet_only_hike
import math

def generate_random_points(n,
                           min_range=4.0,
                           max_range=10.0,
                           min_dist=0.5,
                           max_dist=2.5,
                           max_attempts=50000):

    points = []
    while len(points) < n:
        found_spot = False
        for _ in range(max_attempts):
            # Random candidate in bounding box
            x = random.uniform(min_range, max_range)
            y = random.uniform(min_range, max_range)

            # Check distance constraints relative to existing points
            if all(min_dist <= math.dist((x, y), p) <= max_dist for p in points):
                points.append((x, y))
                found_spot = True
                break  # proceed to place the next point
        
        if not found_spot:
            # If we cannot find a valid point after many tries, stop and inform the user
            raise ValueError(
                f"Could not place point #{len(points)+1} "
                f"within {max_attempts} attempts. "
                "Try decreasing n, lowering min_dist, or increasing max_dist."
            )

    return points

def generate_trajectories(config):
    num_objects = config["num_objects"]
    objects = num_objects * [config["object_type"]]
    if config["use_fixed_locs"]:
        assert len(config["fixed_obj_locs"]) == num_objects, "Number of fixed locations should match number of objects"
        return objects, config["fixed_obj_locs"]
    locations_rel = generate_random_points(num_objects, min_range=config["min_range"], max_range=config["max_range"]) 
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
    parser.add_argument('--config', type=str, default='pybullet_env/configs/whale_run.toml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        config = tomli.load(f)

    # Generate trajectories/CONFIG INITIALIZATION HERE
    objects, locations_rel = generate_trajectories(config)

    formation_type = config["formation_type"]
    # Run simulations
    run_pybullet_only_hike([objects, locations_rel], 
                           output_folder="whale_results", 
                           duration_sec=config['duration_sec'],
                           record_hz=3,
                           num_objects=config["num_objects"],
                           num_drones=int(config["num_agents"])+1,
                           move_whales=config['move_whales'],
                           goal_assignment=config['goal_assignment'],
                           drone_formation_type=formation_type,
                           target_obj=config['object_type'],
                           gnn_model_path=config['gnn_model_path'],
                           search_type=config['search_type'],
                           debug_images=config['debug_images'],
                           )
