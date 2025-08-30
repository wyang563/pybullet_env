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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. helper – repaint would‑be‑transparent pixels to page background
# ------------------------------------------------------------------
def make_black_transparent(image, threshold=10, bg_color=(255, 255, 255)):
    """
    Return an RGBA image in which
      • pixels darker than `threshold` in **all** channels become transparent
      • those pixels' RGB is painted `bg_color` so edge blending has no halo
    """
    image = image.copy()
    h, w = image.shape[:2]

    # alpha channel starts fully opaque
    alpha = np.full((h, w), 255, dtype=np.uint8)

    # mask of "black" pixels (all channels below threshold)
    black_mask = np.all(image < threshold, axis=-1)

    # make them transparent ...
    alpha[black_mask] = 0
    # ... and paint their RGB the same colour as the page background
    image[black_mask] = bg_color

    return np.dstack((image, alpha))


# ------------------------------------------------------------------
# 2. main plotting routine – uses the new helper
# ------------------------------------------------------------------
def plot_boxed_images(img_list, boxes_list, net_corrs,
                      save_path=None, img_data=None,
                      plot_title="Current ICP Iteration"):
    if img_data is None:
        n_images = len(img_list)
    else:
        n_images = len(img_data)

    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
    if n_images == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab20")

    for i in range(n_images):

        # ------------------------------------------------------------------
        # read / convert / add transparency           (<<< changed lines)
        # ------------------------------------------------------------------
        image_bgr = cv2.imread(img_list[i]) if img_data is None else img_data[i]
        if image_bgr is None:
            print(f"Error: Unable to read image {img_list[i]}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgba = make_black_transparent(image_rgb)

        # show the image – no extra α, no interpolation artefacts
        axes[i].imshow(image_rgba, interpolation='nearest')
        axes[i].axis("off")

        # title
        axes[i].set_title("Target Image" if i == 0 else plot_title)

        # ------------------------------------------------------------------
        # draw oriented bounding boxes (unchanged)
        # ------------------------------------------------------------------
        for j, box in enumerate(boxes_list[i]):
            pts = np.array(box).reshape(4, 2)
            color_idx = net_corrs[i][j]
            color = cmap(color_idx % cmap.N + 1)
            face_color = (*color[:3], 0.3)
            poly = patches.Polygon(pts, closed=True,
                                   edgecolor=color,
                                   facecolor=face_color,
                                   linewidth=2)
            axes[i].add_patch(poly)

    plt.tight_layout(pad=0.03, w_pad=0.03, h_pad=0.03, rect=[0, 0, 1, 0.95])

    fig.subplots_adjust(wspace=0, hspace=0)        # remove space between subplots
    fig.patch.set_alpha(0) 
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

def plot_success_rate_histogram(arr, num_agents, save_filename="pybullet_env/icp/histogram.pdf"): # Added save_filename parameter
    """
    Calculates the success rate (ratio of 1s) for each sublist in arr
    and plots the rates as a histogram (bar chart), saving it to a file.

    Args:
        arr: A list of lists, where sublists contain 0s and 1s.
        save_filename (str): The name of the file to save the plot to.
    """
    rates = []
    indices = list(range(len(arr)))

    for i in indices:
        sublist = arr[i]
        sublist_len = len(sublist)
        if sublist_len == 0:
            rates.append(0.0)
        else:
            count_of_ones = sum(1 for item in sublist if item == 1)
            rate = count_of_ones / sublist_len
            rates.append(rate)

    # Create the bar chart (histogram)
    plt.figure(figsize=(10, 6))
    plt.bar(indices, rates, color='skyblue', edgecolor='black')

    plt.xlabel("Index (i)")
    plt.ylabel("Success Rate (Count of 1s / Length)")
    plt.title("Success Rate per Sublist Index")
    plt.xticks(indices) # Ensure each index has a tick
    plt.ylim(0, 1.1) # Set y-axis limits from 0 to 1.1 for clarity
    plt.grid(axis='y', linestyle='--')

    # Save the plot instead of showing it
    save_filename = os.path.join("pybullet_env/icp", f"{num_agents}_histogram.pdf")
    plt.savefig(save_filename, dpi=300)
    print(f"Histogram saved to {save_filename}") # Optional: confirmation message
    plt.close() # Close the plot figure to free memory



def rotate_image(img_file, rot_angle, save_dir, in_img=None, custom_save_file=None, custom_rotation_center=None, save=False):
    # Read the image from the specified directory
    if in_img is None:
        image = cv2.imread(os.path.join(save_dir, img_file))
        if image is None:
            print(f"Error: Unable to read image {os.path.join(save_dir, img_file)}")
            return
    else:
        image = in_img 

    h, w = image.shape[:2]
    
    # Define region dimensions (width x height)
    region_width = w 
    region_height = h 
    
    # Calculate center of the image
    center_x, center_y = w // 2, h // 2
    
    # Determine cropping boundaries for the center region.
    x1 = max(0, center_x - region_width // 2)
    x2 = min(w, center_x + region_width // 2)
    y1 = max(0, center_y - region_height // 2)
    y2 = min(h, center_y + region_height // 2)
    
    # Extract the center region.
    region = image[y1:y2, x1:x2]
    region_h, region_w = region.shape[:2]
    
    # Compute the rotation matrix.
    # OpenCV rotates counter-clockwise for positive angles; if you want clockwise, use -rot_angle
    if custom_rotation_center is not None:
        M = cv2.getRotationMatrix2D(tuple(custom_rotation_center), -rot_angle, 1.0)
    else:
        M = cv2.getRotationMatrix2D((region_w / 2, region_h / 2), -rot_angle, 1.0)
    
    # Rotate the extracted region.
    rotated_region = cv2.warpAffine(region, M, (region_w, region_h))
    
    # Create a copy of the original image and replace the center region with the rotated region.
    new_image = image.copy()

    # You can either:
    # 1) Crop the larger rotated region down to the original region size, or
    # 2) Paste the entire larger region back in a different way, e.g., centered.
    # Example 1: just overwrite with the same bounding box size:
    cropped_rotated = rotated_region[0:region_h, 0:region_w]
    new_image[y1:y2, x1:x2] = cropped_rotated
    
    # Save the rotated image. The filename includes the rotation angle.
    if save:
        if custom_save_file:
            cv2.imwrite(custom_save_file, new_image)
        else:
            save_file = os.path.join(save_dir, f"rotated_{rot_angle}_{img_file}")
            cv2.imwrite(save_file, new_image)
            # print(f"Rotated image saved to {save_file}")
    return new_image

def translate_image(img_file, x, y, save_dir, in_img=None, custom_save_file=None, save=False, vary_height=False, padding=0):
    if in_img is None:
        image = cv2.imread(os.path.join(save_dir, img_file))
        if image is None:
            print(f"Error: Unable to read image {os.path.join(save_dir, img_file)}")
            return
    else:
        image = in_img 

    # vary padding to simulate taking images from different heights
    if not vary_height:
        padding = 0
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # resize image 
    target_size = (2160, 2160)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    
    # Define the dimensions of the center region.
    region_width = w 
    region_height = h 

    # Calculate the center point.
    center_x, center_y = w // 2, h // 2
    
    # Determine cropping boundaries for the center region.
    x1 = max(0, center_x - region_width // 2)
    x2 = min(w, center_x + region_width // 2)
    y1 = max(0, center_y - region_height // 2)
    y2 = min(h, center_y + region_height // 2)
    
    # Extract the center region.
    region = image[y1:y2, x1:x2].copy()
    
    # Create the translation matrix.
    # Note: cv2.warpAffine expects a 2x3 matrix: 
    # [ [1, 0, x_translation], [0, 1, y_translation] ]
    M = np.float32([[1, 0, x], [0, 1, y]])
    
    # Translate the region.
    translated_region = cv2.warpAffine(region, M, (region.shape[1], region.shape[0]))
    
    # Create a copy of the original image and replace the center region with the translated region.
    new_image = image.copy()
    new_image[y1:y2, x1:x2] = translated_region
    if save:
        if custom_save_file:
            cv2.imwrite(custom_save_file, new_image)
        else:
            save_file = os.path.join(save_dir, f"translated_{x}_{y}" + img_file)
            cv2.imwrite(save_file, new_image)
    return new_image 

def shear_image(img_file, shear_angle, save_dir, in_img=None):
    # If no in_img passed in, read from disk
    if in_img is None:
        image_path = os.path.join(save_dir, img_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return
    else:
        image = in_img

    h, w = image.shape[:2]

    # Convert angle (in degrees) to a shear factor
    shear_factor = math.tan(math.radians(shear_angle))

    # We'll expand our output width so the entire sheared image shows.
    # For a horizontal shear:
    new_width = int(w + abs(shear_factor) * h)

    # Build the shear matrix.
    # Note the extra x-translation that shifts the image so it doesn't go negative:
    # If shear_factor > 0, we shift positively; if shear_factor < 0, the shift is 0.
    shift_x = max(0, int(shear_factor * h))
    M = np.float32([
        [1, shear_factor, shift_x],
        [0, 1,           0]
    ])

    # Warp the image into the new bounding box.
    sheered_image = cv2.warpAffine(image, M, (new_width, h))

    out_filename = f"sheered_{shear_angle}_{img_file}"
    out_path = os.path.join(save_dir, out_filename)
    cv2.imwrite(out_path, sheered_image)
    print(f"Sheered image saved to {out_path}")

# def generate_augmentations(img_dir, num_images=5):
#     # CONFIG: TOGGLE FOR DIFFERENT AUGMENTATIONS
#     rotation_angle_range = [-180, 180]
#     translation_range = [0, 0]
#     base_img_path = "base_img.jpg"

#     for _ in range(num_images):
#         # translation = np.random.randint(translation_range[0], translation_range[1], size=2)
#         rotation_angle = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
#         translated_img = translate_image(base_img_path, 0, 0, img_dir)
#         rotate_image(base_img_path, rotation_angle, img_dir, in_img=translated_img)

#     # for img_file in os.listdir(img_dir):
#     #     translation = np.random.randint(translation_range[0], translation_range[1], size=2)
#     #     shear_angle = np.random.randint(shear_angle_range[0], shear_angle_range[1])
#     #     rotation_angle1 = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
#     #     rotation_angle2 = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
#     #     translated_img = translate_image(img_file, translation[0], translation[1], img_dir)
#     #     rotate_image(img_file, rotation_angle1, img_dir, in_img=translated_img)
#     #     rotate_image(img_file, rotation_angle2, img_dir, in_img=translated_img)
#     #     shear_image(img_file, shear_angle, img_dir)

def generate_one_augmentation(img_dir, in_img=None, vary_heights=False):
    rotation_angle_range = [-180, 180]
    translation_range = [-700, 700]
    base_img_path = "base_img.jpg"
    rotation_angle = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
    dx, dy = np.random.randint(translation_range[0], translation_range[1], size=2)
    translated_img = translate_image(base_img_path, dx, dy, img_dir, in_img=in_img, vary_height=vary_heights, save=False)
    return rotate_image(base_img_path, rotation_angle, img_dir, in_img=translated_img, save=False)

def extract_boxes(img_0_path, img_1_path, model_conf_threshold=0.5):
    if not os.path.exists("pybullet_env/icp/icp_plots"):
        os.makedirs("pybullet_env/icp/icp_plots")
    
    model_file = "pybullet_env/icp/whale_data/last.pt"

    model = YOLO(model_file)
    img_0 = cv2.imread(img_0_path)
    img_1 = cv2.imread(img_1_path)

    # Save the images
    cv2.imwrite(f"pybullet_env/icp/img_0.jpg", img_0)
    cv2.imwrite(f"pybullet_env/icp/img_1.jpg", img_1)

    with torch.no_grad():
        img_0_path = f"pybullet_env/icp/img_0.jpg"
        img_1_path = f"pybullet_env/icp/img_1.jpg"
        results = model(source=[img_0_path, img_1_path],
                        conf=model_conf_threshold,
                        imgsz=640,
                        classes=[0],
                        device="cpu",
                        verbose=False,
                    )
        vid1_boxes = results[0].obb.xyxyxyxy.numpy()
        vid2_boxes = results[1].obb.xyxyxyxy.numpy()
        num_boxes1 = vid1_boxes.shape[0]
        num_boxes2 = vid2_boxes.shape[0]

    correct = num_boxes1 == num_boxes2 and min(num_boxes1, num_boxes2) > 0 
    if results[0].obb.conf.numpy().size > 0 and results[1].obb.conf.numpy().size > 0:
        lowest_conf = min(results[0].obb.conf.numpy().min(),
                        results[1].obb.conf.numpy().min())
    else:
        lowest_conf = None
    num_whales = max(num_boxes1, num_boxes2)
    return vid1_boxes, vid2_boxes, correct, num_whales, lowest_conf

def calculate_box_plots(files):
    if not os.path.exists("pybullet_env/icp/icp_plots"):
        os.makedirs("pybullet_env/icp/icp_plots")
    
    model_file = "pybullet_env/icp/whale_data/last.pt"

    model = YOLO(model_file)
    correlations = []
    img_list = []
    boxes = []
    for i in range(1, len(files)):
        img_1_path = files[0]
        img_2_path = files[i]
        img_1 = cv2.imread(img_1_path)
        height, width = img_1.shape[:2]
        img_2 = cv2.imread(img_2_path)

        # Crop the center 2000x2000 portion

        # crop_size = 2150
        # center_x, center_y = width // 2, height // 2
        # x1 = max(0, center_x - crop_size // 2)
        # y1 = max(0, center_y - crop_size // 2)
        # x2 = min(width, center_x + crop_size // 2)
        # y2 = min(height, center_y + crop_size // 2)
        # img_1 = img_1[y1:y2]
        # img_2 = img_2[y1:y2] 

        # Save the images
        cv2.imwrite(f"pybullet_env/icp/img_0.jpg", img_1)
        cv2.imwrite(f"pybullet_env/icp/img_{i}.jpg", img_2)

        with torch.no_grad():
            img_1_path = f"pybullet_env/icp/img_0.jpg"
            img_2_path = f"pybullet_env/icp/img_{i}.jpg"
            results = model(source=[img_1_path, img_2_path],
                            conf=0.50,
                            imgsz=640,
                            classes=[0],
                            device="cpu",
                        )
            vid1_boxes = results[0].obb.xyxyxyxy.numpy()
            vid2_boxes = results[1].obb.xyxyxyxy.numpy()
            num_boxes1 = vid1_boxes.shape[0]
            num_boxes2 = vid2_boxes.shape[0]

            # get rid of least confident predictions until number of boxes match
            diff = abs(num_boxes1 - num_boxes2)
            assert diff == 0, "Number of boxes in images do not match"

            # if num_boxes1 > num_boxes2:
            #     vid1_boxes = vid1_boxes[:-diff]
            # elif num_boxes2 > num_boxes1:
            #     vid2_boxes = vid2_boxes[:-diff]

        # get centers of boxes
        # vid1_centers = np.mean(vid1_boxes, axis=1)
        # vid2_centers = np.mean(vid2_boxes, axis=1)

        _, corr, _ = rot_icp(vid2_boxes, vid1_boxes, use_point=False)
        if i == 1:
            img_list.append(img_1_path)
            boxes.append(vid1_boxes)
        img_list.append(img_2_path)
        boxes.append(vid2_boxes)
        correlations.append(corr.tolist())

    # plot results
    correlations.insert(0, [i for i in range(len(correlations[0]))])
    print(correlations)
    plot_boxed_images(img_list, boxes, correlations, save_path=f"pybullet_env/icp/whale_data/box_plot.pdf")

def find_num_boxes(img_dir):
    num_whales = []
    wrong_images = []
    model = YOLO("pybullet_env/icp/whale_data/last.pt")
    for f in os.listdir(img_dir):
        file_path = os.path.join(img_dir, f)
        results = model(source=file_path,
                            conf=0.60,
                            imgsz=640,
                            classes=[0],
                            device="cpu",
                        ) 
        vid1_boxes = results[0].obb.xyxyxyxy.numpy()
        if len(vid1_boxes) != 9:
            wrong_images.append(f)
        num_whales.append(len(vid1_boxes))
    
    print(wrong_images)

    # Plot a histogram of the number of whales detected
    plt.figure(figsize=(6,6))
    # Use at least 1 bin; if all zeros, default to 1 bin
    if len(num_whales) > 0:
        min_val = min(num_whales)
        max_val = max(num_whales)
        bins = range(min_val, max_val + 2) if min_val < max_val else 1
    else:
        bins = 1
    
    plt.hist(num_whales, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Number of Whales Detected')
    plt.ylabel('Frequency')
    plt.title('Distribution of Whale Detections')
    
    # Save the histogram in the same directory or elsewhere
    hist_path = os.path.join("pybullet_env/icp/num_whales_hist.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogram of whale detections saved to {hist_path}")
    

def pairwise_box_plots(img_dir):
    os.makedirs("pybullet_env/icp/whale_data/icp_experiments/shot1_box_plots", exist_ok=True)
    shot_files = os.listdir(img_dir)
    failed_boxes = 0
    total = 0
    for i in range(len(shot_files)):
        for j in range(len(shot_files)):
            if i != j:
                img_files = []
                img_files.append(f"pybullet_env/icp/whale_data/icp_experiments/shot1/{shot_files[i]}")
                img_files.append(f"pybullet_env/icp/whale_data/icp_experiments/shot1/{shot_files[j]}")
                try:
                    calculate_box_plots(img_files)
                except:
                    print("number of boxes did not match!")
                    failed_boxes += 1
            total += 1
    print(failed_boxes)
    print(total)

def eval_whale_model(model_conf_threshold=0.3, num_augmentations=100, blur_factor=0):
    """
    Evaluates the YOLO model's accuracy in detecting the correct number of whales.
    It processes images from a specified directory, generates augmentations for each,
    runs the model, and compares detected boxes against the expected number from the filename.

    Args:
        model_conf_threshold (float): Confidence threshold for YOLO predictions.
        num_augmentations (int): Number of augmentations to generate per base image.
    """
    input_dir = "pybullet_env/icp/whale_data/yolo_dataset/images/val"
    temp_dir = "pybullet_env/icp/whale_data/whale_icp_eval"
    output_dir = "pybullet_env/icp/whale_data/eval_data_json"
    output_filename = os.path.join(output_dir, f"model_accuracy_conf_{model_conf_threshold}_blur_{blur_factor}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # results[num_whales] = {'correct': 0, 'total': 0}
    results_agg = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Load YOLO model
    model_file = "pybullet_env/icp/whale_data/best.pt"
    model = YOLO(model_file)

    # Regex to parse filename
    filename_pattern = re.compile(r"(\d+)_whale_(\d+)\.png")

    base_image_files = [f for f in os.listdir(input_dir) if filename_pattern.match(f)]
    
    print(f"Starting model evaluation with conf={model_conf_threshold}...")
    for base_filename in base_image_files:
        match = filename_pattern.match(base_filename)
        if not match:
            print(f"Warning: Skipping file with unexpected name format: {base_filename}")
            continue
            
        expected_num_whales = int(match.group(1))
        base_img_path_src = os.path.join(input_dir, base_filename)
        
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
        base_img = cv2.GaussianBlur(base_img, (blur_factor, blur_factor), 0)
        cv2.imwrite(base_img_path_dst, base_img)
        
        # Generate augmentations
        for _ in tqdm(range(num_augmentations), desc=f"Processing {base_filename}"):
            augmented_img = generate_one_augmentation(temp_dir)
            try:
                with torch.no_grad():
                    yolo_results = model(source=augmented_img,
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
                    
                    results_agg[expected_num_whales]['total'] += 1
                    if detected_num_boxes == expected_num_whales:
                        results_agg[expected_num_whales]['correct'] += 1

            except Exception as e:
                print(f"Error during YOLO prediction for files related to {base_filename}: {e}")


    # Calculate final accuracies
    final_results = {}
    for num_whales, counts in results_agg.items():
        total = counts['total']
        correct = counts['correct']
        accuracy = (correct / total) if total > 0 else 0
        final_results[num_whales] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
        print(f"Whales: {num_whales}, Correct: {correct}, Total: {total}, Accuracy: {accuracy:.4f}")

    # Save results to JSON
    try:
        # Ensure keys are strings for JSON
        final_results_serializable = {str(k): v for k, v in final_results.items()}
        with open(output_filename, "w") as f:
            json.dump(final_results_serializable, f, indent=2)
        print(f"Model evaluation results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


def eval_whale_icp(num_base_images_per_agent_count=10, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=False, blur_factor=0, debug_failed_icp=False):
    """
    Evaluates ICP accuracy across varying numbers of agents (2, 4, 8, 16).
    For each agent count, it selects random base images, generates augmentations,
    checks YOLO box count consistency, and then runs cyclical ICP if consistent.
    Uses a fixed YOLO confidence threshold of 0.3.

    Args:
        num_base_images_per_agent_count (int): Number of random base images to test for each agent count.
    """
    img_dir = "pybullet_env/icp/whale_data/final_model_eval"
    temp_dir = "pybullet_env/icp/whale_data/whale_icp_eval"
    output_dir = "pybullet_env/icp/whale_data/eval_data_json"
    output_filename = os.path.join(output_dir, f"icp_accuracy_vs_agents_conf_{model_conf_threshold}_vary_heights_{vary_heights}_use_point_icp_{use_point_icp}_blur_{blur_factor}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    debug_plot_dir = os.path.join(output_dir, "icp_failed_debug_plots")
    if debug_failed_icp:
        os.makedirs(debug_plot_dir, exist_ok=True)


    # agent_counts = [2, 4, 8, 16]
    agent_counts = [2, 4, 6, 8, 10, 12, 14, 16]
    
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
            if blur_factor > 0:
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

                # try:
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

                # except Exception as e:
                #     print(f"Error during ICP for {base_filename} with {num_agents} agents: {e}")
                #     # Optionally record this as a failure? Depends on desired analysis.
                #     overall_results[num_agents][expected_num_whales].append(False) 
                if not icp_correct and debug_failed_icp:
                    print("PLOTTING FAILED RUN")
                    plot_title = f"DEBUG_ICP_FAIL_{os.path.splitext(base_filename)[0]}_agents_{num_agents}"
                    initial_boxes = icp_boxes_list[0]
                    if initial_boxes is not None and len(initial_boxes) > 0:
                        plot_rectangles(
                            pc_name=plot_title,
                            cloud1=initial_boxes,
                            cloud2=initial_boxes, # Plotting correspondence of initial boxes to themselves
                            correspondences=final_net_corr,
                            rot_angle=0,
                            save=True,
                            save_dir=debug_plot_dir
                        )
                        print(f"Saved debug ICP plot: {os.path.join(debug_plot_dir, plot_title + '_registration.png')}")
                    else:
                        print(f"Skipping debug plot for {base_filename} with {num_agents} agents due to no initial boxes.")

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

def eval_icp_points(num_runs=1, noise_level=20, height_variation=0.5, record=False):
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
        
    for _ in range(num_runs):
        for base_filename in tqdm(all_label_files):
            # Select a random file
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

            # get set of random transforms
            all_points_transforms = [all_points] 
            for _ in range(15):
                random_translation = np.random.randint(-1000, 1000, size=2)
                rot_theta = np.deg2rad(np.random.randint(-180, 180))
                R = np.array([[np.cos(rot_theta), -np.sin(rot_theta)], [np.sin(rot_theta), np.cos(rot_theta)]])
                transformed_points = []

                # apply rotations
                for box in all_points:
                    transformed_box = []
                    for point in box:
                        point_array = np.array(point)
                        transformed_point = np.dot(R, point_array)
                        transformed_box.append(transformed_point.tolist())
                    transformed_points.append(transformed_box)

                transformed_points = np.array(transformed_points)

                # apply noise and apply heigh variation            
                height_change_constant = np.random.uniform(1 - height_variation, 1)
                scale_direction = random.choice([-1, 1])
                if scale_direction == -1:
                    height_change_constant = 1 / height_change_constant 
                noise = np.random.normal(0, noise_level, all_points.shape)
                transformed_points = transformed_points + noise
                transformed_points = transformed_points * height_change_constant 

                # apply translations
                transformed_points = transformed_points + random_translation
                all_points_transforms.append(transformed_points)

            for num_agents in [2, 4, 8, 16]:
                cyclical_icp_points = all_points_transforms[:num_agents]
                # cyclical ICP 
                correlations = []
                for i in range(num_agents):
                    boxes_0 = cyclical_icp_points[i]
                    boxes_1 = cyclical_icp_points[(i + 1) % num_agents]
                    
                    # Ensure boxes are not None before ICP
                    if boxes_0 is None or boxes_1 is None:
                        raise ValueError("Cannot perform ICP with None boxes.")

                    if i == 0 and record:
                        _, corr, _, recorded_clouds = rot_icp(boxes_1, boxes_0, use_point=False, record=True)     
                        # plot recorded cloud
                        for sample_rot_theta, _, _, iter_correlations, point_clouds in recorded_clouds:
                            for cloud_idx, cloud in enumerate(point_clouds):
                                corr = iter_correlations[cloud_idx]
                                plot_rectangles(f"Noise Level rot theta {sample_rot_theta} index {cloud_idx}", boxes_0, cloud, corr, sample_rot_theta)
                    else:
                        _, corr, _ = rot_icp(boxes_1, boxes_0, use_point=False, record=False)
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
                overall_results[num_agents][num_whales].append(icp_correct)
    
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

def plot_pair_boxes(boxes0, boxes1, corr, save_path, img_data):
    """
    Plots two images side by side with box polygons overlaid.
    
    Args:
        boxes0 (np.ndarray): Array of shape (N, 4, 2) for scene 0.
        boxes1 (np.ndarray): Array of shape (N, 4, 2) for scene 1.
        corr (list or np.ndarray): Mapping from boxes0 indices to boxes1 indices.
        save_path (str): File path to save the final plot.
        img_data (list): List of two cv2 image arrays.
    """
    import matplotlib.pyplot as plt
    import cv2
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Scene 0: draw each box polygon and label with its index
    ax0 = axes[0]
    img0_rgb = cv2.cvtColor(img_data[0], cv2.COLOR_BGR2RGB)
    ax0.imshow(img0_rgb)
    for i, box in enumerate(boxes0):
        centroid = box.mean(axis=0)
        polygon = plt.Polygon(box, fill=None, edgecolor='red', lw=2)
        ax0.add_patch(polygon)
        ax0.text(centroid[0], centroid[1], f"{i}", color='white', fontsize=12,
                 ha='center', va='center')
    ax0.set_title("Scene 0")
    ax0.axis("off")
    
    # Plot Scene 1: use corr to map each box in boxes0 to the corresponding box in boxes1
    ax1 = axes[1]
    img1_rgb = cv2.cvtColor(img_data[1], cv2.COLOR_BGR2RGB)
    ax1.imshow(img1_rgb)
    for i, box_index in enumerate(corr):
        box = boxes1[box_index]
        centroid = box.mean(axis=0)
        polygon = plt.Polygon(box, fill=None, edgecolor='red', lw=2)
        ax1.add_patch(polygon)
        ax1.text(centroid[0], centroid[1], f"{i}", color='white', fontsize=12,
                 ha='center', va='center')
    ax1.set_title("Scene 1")
    ax1.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("STARTING PROGRAM")
    # eval_whale_model(model_conf_threshold=0.3, num_augmentations=100, blur_factor=35)
    # eval_whale_icp(100, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=False, blur_factor=0) 
    eval_whale_icp(100, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=True, blur_factor=0) 

    # for blur_factor in [0, 5, 15, 25, 35]:
    #     eval_whale_icp(100, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=True, blur_factor=blur_factor) 
    #     eval_whale_icp(100, model_conf_threshold=0.3, vary_heights=False, record=False, use_point_icp=False, blur_factor=blur_factor) 
    # eval_icp_points(noise_level=60, height_variation=0.5, record=True)
    # for noise_level in [0, 20, 40, 60, 100]:
    #     eval_icp_points(noise_level=noise_level, height_variation=0.5)

    # for height_variation in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
    #     eval_icp_points(noise_level=0, height_variation=height_variation)


