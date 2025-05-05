import os
import re
import random
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from icp import rot_icp # Assuming rot_icp is in a local 'icp' module

def eval_icp_points(num_runs=50, noise_level=20, height_variation=0.5):
    """
    Loads the vertex coordinates for all rectangles from a randomly selected 
    label file in the specified directory.
    """
    label_dir = "pybullet_env/icp/whale_data/final_model_eval_labels"
    output_dir = "pybullet_env/icp/whale_data/eval_data_json"
    all_label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    if not all_label_files:
        print(f"Error: No label files found in {label_dir}")
        return None

    filename_pattern = re.compile(r"(\d+)_whale_(\d+)\.txt")
    overall_results = defaultdict(lambda: defaultdict(list))
        
    for _ in tqdm(range(num_runs)):
        # Select a random file
        base_filename = random.choice(all_label_files)
        file_path = os.path.join(label_dir, base_filename)
        match = filename_pattern.match(base_filename)
        if not match: continue # Should not happen due to pre-filtering
        num_whales = int(match.group(1))

        rectangles = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9: # Expecting class label + 8 coordinates
                    try:
                        # Extract coordinates, convert to float, and reshape
                        coords = np.array([float(p) for p in parts[1:]])
                        rectangle_points = coords.reshape((4, 2))
                        rectangles.append(rectangle_points)
                    except ValueError:
                        print(f"Warning: Could not parse line in {base_filename}: {line.strip()}")
                else:
                        print(f"Warning: Skipping malformed line in {base_filename}: {line.strip()}")

        if not rectangles:
            print(f"Warning: No valid rectangles found in {base_filename}")
            return None

        # Stack the list of (4, 2) arrays into a single (N, 4, 2) array and center points around origin
        all_points = np.stack(rectangles)
        all_points = 2160 * all_points
        global_center = np.mean(all_points.reshape(-1, 2), axis=0)
        all_points = all_points - global_center

        # Pre-allocate array for transforms
        num_transforms_to_generate = 16 # 1 original + 15 transformed
        all_points_transforms = np.zeros((num_transforms_to_generate, *all_points.shape))
        all_points_transforms[0] = all_points

        # Generate transforms using vectorized operations
        for i in range(1, num_transforms_to_generate):
            random_translation = np.random.randint(-1000, 1000, size=2)
            rot_theta = np.deg2rad(np.random.randint(-180, 180))
            R = np.array([[np.cos(rot_theta), -np.sin(rot_theta)], 
                          [np.sin(rot_theta), np.cos(rot_theta)]])
            
            # Apply rotation using einsum: 'ij,nkj->nki' 
            # This efficiently applies the 2x2 matrix R to each 2-element point vector
            # across all N boxes and 4 points per box.
            rotated_points = np.einsum('ij,nkj->nki', R, all_points_transforms[0], optimize=True) 

            # apply noise and apply height variation            
            height_change_constant = np.random.uniform(1 - height_variation, 1)
            scale_direction = random.choice([-1, 1])
            if scale_direction == -1:
                 # Avoid division by zero if height_variation is 1
                if height_change_constant == 0: 
                    height_change_constant = np.random.uniform(1e-6, 1) # Use a small value instead
                height_change_constant = 1 / height_change_constant 
            noise = np.random.normal(0, noise_level, all_points.shape)
            
            # Apply noise first, then scaling
            noisy_rotated_points = rotated_points + noise
            scaled_noisy_rotated_points = noisy_rotated_points * height_change_constant 

            # apply translations
            transformed_points = scaled_noisy_rotated_points + random_translation
            all_points_transforms[i] = transformed_points # Store the final transformed points

        for num_agents in [2, 4, 8, 16]:
            # Slice the pre-computed transforms
            cyclical_icp_points = all_points_transforms[:num_agents] 
            # cyclical ICP 
            correlations = []
            # Add try-except for robustness during ICP/correlation calculation
            try:
                for i in range(num_agents):
                    boxes_0 = cyclical_icp_points[i]
                    boxes_1 = cyclical_icp_points[(i + 1) % num_agents]
                    
                    # Ensure boxes are numpy arrays and not empty
                    if not isinstance(boxes_0, np.ndarray) or boxes_0.size == 0 or \
                       not isinstance(boxes_1, np.ndarray) or boxes_1.size == 0:
                         raise ValueError("Cannot perform ICP with invalid box data.")

                    _, corr, _ = rot_icp(boxes_1, boxes_0, use_point=False)
                    # Check if corr is valid before converting to list
                    if corr is None or not hasattr(corr, 'tolist'):
                         raise ValueError("ICP returned invalid correlation.")
                    correlations.append(corr.tolist())
                
                # Calculate net correspondence 
                net_corrs = []
                # Check if correlations list is empty or first element is empty
                if not correlations or not correlations[0]:
                     # If no correlations, assume failure or handle as appropriate
                     print(f"Warning: No correlations generated for {base_filename}, {num_agents} agents.")
                     overall_results[num_agents][num_whales].append(False)
                     continue # Skip to next num_agents

                num_corr_points = len(correlations[0])
                if num_corr_points == 0:
                    # If correlation length is 0, assume failure
                    print(f"Warning: Zero-length correlation for {base_filename}, {num_agents} agents.")
                    overall_results[num_agents][num_whales].append(False)
                    continue # Skip to next num_agents

                for i in range(len(correlations)):
                    composite = np.arange(num_corr_points) 
                    for j in range(i, -1, -1):
                        current_corr = correlations[j]
                        # Basic validation of correlation indices
                        if not all(isinstance(idx, int) and 0 <= idx < num_corr_points for idx in current_corr):
                             raise ValueError(f"Invalid index or type in correlation {j}: {current_corr}")

                        new_composite = np.zeros(num_corr_points, dtype=int)
                        # Check bounds before assignment
                        if len(current_corr) != num_corr_points:
                             raise ValueError(f"Correlation length mismatch at step {j}.")
                        
                        # Vectorized assignment is safer if indices are guaranteed unique,
                        # but loop is clearer for potentially non-unique indices from ICP.
                        for k in range(num_corr_points):
                             new_composite[current_corr[k]] = composite[k]
                        composite = new_composite.copy()
                    net_corrs.append(composite)

                # The final net correspondence after all steps (0 to num_agents-1)
                final_net_corr = net_corrs[-1] 
                
                # Check if it's the identity permutation
                icp_correct = final_net_corr.tolist() == list(range(len(final_net_corr)))
                overall_results[num_agents][num_whales].append(icp_correct)

            except Exception as e: # Catch potential errors during ICP/correlation processing
                print(f"Error during ICP/Correlation processing for {base_filename} with {num_agents} agents: {e}")
                overall_results[num_agents][num_whales].append(False) # Record as failure
    
    # save aggregated results to JSON
    output_filename = os.path.join(output_dir, f"icp_point_test_noiselevel_{noise_level}_heightvar_{height_variation}.json")
    try:
        # Convert inner defaultdict to dict for JSON serialization
        serializable_results = {
            str(agents): {str(whales): results for whales, results in whale_dict.items()}
            for agents, whale_dict in overall_results.items()
        }
        with open(output_filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nICP evaluation results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving aggregated results to JSON: {e}")

if __name__ == "__main__":
    print("STARTING PROGRAM")
    eval_icp_points(num_runs=50, noise_level=100, height_variation=0.7)