import torch
from datetime import datetime
from ultralytics import YOLO
from icp import rot_icp, plot_rectangles 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import math
import random
import json
from collections import defaultdict
from tqdm import tqdm
import re
import glob
from whale_icp_test import generate_one_augmentation, rotate_image, plot_pair_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_whale_model(model_conf_threshold=0.3, blur_factor=0):
    """
    Evaluates the YOLO model's accuracy in detecting the correct number of whales.
    It processes images from a specified directory, generates augmentations for each,
    runs the model, and compares detected boxes against the expected number from the filename.

    Args:
        model_conf_threshold (float): Confidence threshold for YOLO predictions.
        num_augmentations (int): Number of augmentations to generate per base image.
    """
    input_dir = "pybullet_env/icp/whale_data/yolo_dataset/images/val"
    input_label_dir = "pybullet_env/icp/whale_data/yolo_dataset/labels/val"
    temp_dir = "pybullet_env/icp/whale_data/whale_icp_eval"
    output_dir = "pybullet_env/icp/whale_data/eval_data_json"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # results[num_whales] = {'correct': 0, 'total': 0}
    correct = 0
    total = 0
    
    # Load YOLO model
    model_file = "pybullet_env/icp/whale_data/best.pt"
    model = YOLO(model_file)

    # Regex to parse filename
    print(f"Starting model evaluation with conf={model_conf_threshold}...")
    for base_filename in os.listdir(input_dir):
        base_img_path_src = os.path.join(input_dir, base_filename)
        label_file_path = os.path.join(input_label_dir, base_filename.replace(".jpg", ".txt"))
        
        # --- Prepare Augmentations ---
        # Clear temp directory
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
            
        # Copy base image to temp dir as base_img.jpg
        base_img_path_dst = os.path.join(temp_dir, "base_img.jpg")
        base_img = cv2.imread(base_img_path_src)
        if base_img is None:
            print(f"Error reading base image: {base_img_path_src}")
            continue

        # Blur base image
        if blur_factor > 0 and blur_factor % 2 == 1:
            base_img = cv2.GaussianBlur(base_img, (blur_factor, blur_factor), 0)
        cv2.imwrite(base_img_path_dst, base_img)
        
        try:
            with torch.no_grad():
                yolo_results = model(source=base_img,
                                    conf=model_conf_threshold,
                                    imgsz=640,
                                    classes=[0],
                                    device=device,
                                    verbose=False,
                                    stream=False) # Process as batch if stream=False

            for result in yolo_results:
                detected_num_boxes = 0
                if result.obb is not None and result.obb.xyxyxyxy is not None:
                    detected_num_boxes = len(result.obb.xyxyxyxy.cpu().numpy())
                # get expected number of whales
                with open(label_file_path, 'r') as f:
                    label_data = f.readlines()
                expected_num_whales = len(label_data)
                # if expected_num_whales == detected_num_boxes:
                #     correct += 1
                # total += 1
                total += expected_num_whales
                correct += detected_num_boxes

        except Exception as e:
            print(f"Error during YOLO prediction for files related to {base_filename}: {e}")

    # Calculate final accuracies
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

def eval_whale_icp(num_base_images_per_agent_count=10, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=False, blur_factor=0):
    """
    Evaluates ICP accuracy across varying numbers of agents (2, 4, 8, 16).
    For each agent count, it selects random base images, generates augmentations,
    checks YOLO box count consistency, and then runs cyclical ICP if consistent.
    Uses a fixed YOLO confidence threshold of 0.3.

    Args:
        num_base_images_per_agent_count (int): Number of random base images to test for each agent count.
    """
    img_dir = "pybullet_env/icp/whale_data/final_model_eval"
    label_dir = "pybullet_env/icp/whale_data/yolo_dataset/labels/val"
    temp_dir = "pybullet_env/icp/whale_data/whale_icp_eval"
    output_dir = "pybullet_env/icp/whale_data/eval_data_json"
    output_filename = os.path.join(output_dir, f"icp_accuracy_vs_agents_conf_{model_conf_threshold}_vary_heights_{vary_heights}_use_point_icp_{use_point_icp}_blur_{blur_factor}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    agent_counts = [2, 4, 8, 16]
    
    # overall_results[num_agents][num_whales] = [list of booleans indicating ICP success]
    overall_results = defaultdict(lambda: defaultdict(list))
    
    # Load YOLO model
    model_file = "pybullet_env/icp/whale_data/best.pt"
    model = YOLO(model_file)
    
    # Regex to parse filename
    filename_pattern = re.compile(r"(\d+)_whale_(\d+)\.png")
    
    all_base_image_files = [f for f in os.listdir(img_dir) if filename_pattern.match(f)]
    if not all_base_image_files:
        print(f"Error: No base images found in {img_dir}")
        return

    print(f"Starting ICP evaluation with conf={model_conf_threshold}...")
    
    for num_agents in agent_counts:
        print(f"\n--- Evaluating with {num_agents} agents ---")
        
        # Select random base images for this agent count
        selected_base_files = random.sample(all_base_image_files, min(num_base_images_per_agent_count, len(all_base_image_files)))
        
        for base_filename in tqdm(selected_base_files, desc=f"Agent Count {num_agents}"):
            match = filename_pattern.match(base_filename)
            if not match: continue # Should not happen due to pre-filtering
            
            expected_num_whales = int(match.group(1))
            base_img_path_src = os.path.join(img_dir, base_filename)

            # --- Prepare Augmentations ---
            # Clear temp directory
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            
            # Copy base image
            base_img_path_dst = os.path.join(temp_dir, "base_img.jpg")
            base_img = cv2.imread(base_img_path_src)
            if base_img is None: 
                continue

            # Blur base image
            if blur_factor > 0 and blur_factor % 2 == 1:
                base_img = cv2.GaussianBlur(base_img, (blur_factor, blur_factor), 0)
            cv2.imwrite(base_img_path_dst, base_img)

            # Generate augmentations and store paths/results
            augmented_data = [] # List to store {'path': path, 'boxes': boxes, 'box_count': count}
            images = []
            all_counts_match = True
            
            for n in range(num_agents):
                aug_filename = f"aug_{n}.png"
                aug_filepath = os.path.join(temp_dir, aug_filename)
                # Generate one augmentation and save it
                augmented_img = generate_one_augmentation(temp_dir, in_img=base_img, vary_heights=vary_heights) # Assumes this returns the image array
                if record:
                    images.append(augmented_img)

                # Run YOLO on the single augmented image
                try:
                    with torch.no_grad():
                        yolo_result = model(source=augmented_img,
                                            conf=model_conf_threshold,
                                            imgsz=640,
                                            classes=[0],
                                            device=device,
                                            verbose=False)[0] # Get the first (only) result

                    detected_boxes = None
                    detected_count = 0
                    if yolo_result.obb is not None and yolo_result.obb.xyxyxyxy is not None:
                        detected_boxes = yolo_result.obb.xyxyxyxy.cpu().numpy()
                        detected_count = len(detected_boxes)

                    augmented_data.append({'path': aug_filepath, 'boxes': detected_boxes, 'box_count': detected_count})

                    if detected_count != expected_num_whales:
                        all_counts_match = False
                        # print(f"Mismatch: Expected {expected_num_whales}, got {detected_count} for {aug_filepath}")
                        break # No need to check further augmentations for this base image

                except Exception as e:
                    print(f"Error during YOLO prediction for image number {n}: {e}")
                    all_counts_match = False
                    break
            
            # --- Run Cyclical ICP if counts matched ---
            if all_counts_match and expected_num_whales > 0:
                correlations = []
                icp_boxes_list = [data['boxes'] for data in augmented_data]

                try:
                    for i in range(num_agents):
                        boxes_0 = icp_boxes_list[i]
                        boxes_1 = icp_boxes_list[(i + 1) % num_agents]
                        
                        # Ensure boxes are not None before ICP
                        if boxes_0 is None or boxes_1 is None:
                                raise ValueError("Cannot perform ICP with None boxes.")

                        if not record:
                            _, corr, _ = rot_icp(boxes_1, boxes_0, use_point=use_point_icp)
                        else:
                            _, corr, _, recorded_clouds = rot_icp(boxes_1, boxes_0, use_point=use_point_icp, record=True)

                            # plot images of rotations
                            img0 = images[i]
                            img1 = images[(i + 1) % num_agents]
                            # print("RECORDED CLOUDS")
                            # print(recorded_clouds)
                            for rot_theta, global_center, transforms, _, point_clouds in enumerate(recorded_clouds):
                                # calculate net transforms we need to perform
                                rot_theta_degs = np.rad2deg(rot_theta)
                                # rotate image 1
                                rotated_img1 = rotate_image(base_filename, rot_theta_degs, temp_dir, in_img=img1, save=False)
                                boxes0 = icp_boxes_list[i]
                                boxes1 = point_clouds[-1] 
                                # print(boxes0)
                                # print(boxes1)
                                # convert boxes1 coordinates to cv2 image coordinates
                                boxes1 = boxes1 + global_center
                                plot_pair_boxes(boxes0, boxes1, corr, f"pybullet_env/icp/whale_data/icp_whale_height_variation/{base_filename}_{rot_theta_degs}.png", [img0, rotated_img1])

                            assert False
                        correlations.append(corr.tolist())

                    # Calculate net correspondence 
                    net_corrs = []
                    for i in range(len(correlations)):
                        composite = np.arange(len(correlations[i]))
                        for j in range(i, -1, -1):
                            new_composite = np.zeros(len(composite), dtype=int)
                            for k in range(len(composite)):
                                new_composite[correlations[j][k]] = composite[k]
                            composite = new_composite.copy()
                        net_corrs.append(composite)

                    # The final net correspondence after all steps (0 to num_agents-1)
                    final_net_corr = net_corrs[-1] 
                    
                    # Check if it's the identity permutation
                    icp_correct = final_net_corr.tolist() == list(range(len(final_net_corr)))
                    overall_results[num_agents][expected_num_whales].append(icp_correct)

                except Exception as e:
                    print(f"Error during ICP for {base_filename} with {num_agents} agents: {e}")
                    # Optionally record this as a failure? Depends on desired analysis.
                    overall_results[num_agents][expected_num_whales].append(False) 

            elif not all_counts_match:
                 # Record failure if box counts didn't match? Or just skip? Skipping for now.
                 pass


    # Save aggregated results to JSON
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

def plot_blur_accuracy():
    """
    Plots blurred images side by side with their corresponding accuracy values as captions.
    Removes black padding from images and saves the plot as a PDF file.
    """
    # Data points
    blur_factors = [5, 15, 25, 35]
    accuracies = [0.9712, 0.9317, 0.7941, 0.5277]
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Create a grid for the images
    img_grid = plt.GridSpec(1, 4, figure=fig)
    img_grid.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3)
    
    # Load and display images
    for idx, blur_factor in enumerate(blur_factors):
        img_path = f"pybullet_env/icp/whale_data/blur_{blur_factor}.jpg"
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            # Convert to grayscale for thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find non-black rows (where mean intensity > threshold)
            row_means = np.mean(gray, axis=1)
            threshold = 10  # Adjust this threshold if needed
            non_black_rows = row_means > threshold
            
            # Find the first and last non-black rows
            if np.any(non_black_rows):
                first_non_black = np.where(non_black_rows)[0][0]
                last_non_black = np.where(non_black_rows)[0][-1]
                
                # Crop the image to remove black padding
                img = img[first_non_black:last_non_black+1, :]
            
            # Convert BGR to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create subplot and display image
            ax = fig.add_subplot(img_grid[0, idx])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Blur: {blur_factor}\nAccuracy: {accuracies[idx]:.4f}', 
                        fontsize=12, pad=10)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('pybullet_env/icp/whale_data/blur_accuracy_plot.pdf', 
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # data: 35 blur (0.5277), 25 blur (0.7941), 15 blur (0.9317), 5 blur (0.9262) 
    # plot_blur_accuracy()
    # eval_whale_model(model_conf_threshold=0.3, blur_factor=5)
    for blur_factor in [0, 5, 15, 25, 35]:
        eval_whale_icp(num_base_images_per_agent_count=50, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=False, blur_factor=blur_factor)
        eval_whale_icp(num_base_images_per_agent_count=50, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=True, blur_factor=blur_factor)
        
