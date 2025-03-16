import torch
from datetime import datetime
from ultralytics import YOLO
from icp import rot_icp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_black_transparent(image, threshold=10):
    # Create an alpha channel set to 255 (opaque)
    alpha = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Create a mask where pixels are considered black (all channels are below the threshold)
    black_mask = np.all(image < threshold, axis=-1)
    alpha[black_mask] = 0

    # Combine the original image with the alpha channel to form an RGBA image.
    image_rgba = np.dstack((image, alpha))
    return image_rgba

def plot_boxed_images(img_list, boxes_list, net_corrs, save_path=None):
    """
    Reads in a list of image paths, draws the oriented bounding boxes from the corresponding boxes_list
    on each image, labels each box using the corresponding net_corrs mapping, and displays the images
    side by side (left to right) in one plot.
    
    Args:
        img_list (list[str]): List of file paths for the images.
        boxes_list (list[np.ndarray]): List where each element is an array of bounding boxes for the corresponding image.
            Each bounding box is expected to be in a format that can be reshaped into 4 points (4,2).
        net_corrs (list[np.ndarray or list]): List where net_corrs[i][j] gives the label for the jth box in the ith image.
        save_path (str): Optional. File path to save the resulting plot.
    """

    # Load the background image (assumed to be in the current working directory)
    bg = cv2.imread("pybullet_env/icp/whale_data/blank_ocean.jpg")
    if bg is None:
        print("Error: Unable to load background.jpg")
        return
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg_h, bg_w = bg.shape[:2]
    crop_width = 1920
    crop_height = 1920
    center_x, center_y = bg_w // 2, bg_h // 2
    x1 = max(0, center_x - crop_width // 2)
    x2 = min(bg_w, center_x + crop_width // 2)
    y1 = max(0, center_y - crop_height // 2)
    y2 = min(bg_h, center_y + crop_height // 2)
    bg = bg[y1:y2, x1:x2]

    n_images = len(img_list)
    _, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
    # If only one image, make axes iterable.
    if n_images == 1:
        axes = [axes]


    # Create a colormap that will generate a unique color for each box index.
    cmap = plt.cm.get_cmap("tab20")

    for i in range(n_images):
        # First, display the background image.
        axes[i].imshow(bg)
        # Read and convert the image from BGR to RGB.
        image = cv2.imread(img_list[i])
        if image is None:
            print(f"Error: Unable to read image {img_list[i]}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = make_black_transparent(image) 
        # Overlay the actual image with a slight transparency (optional).
        axes[i].imshow(image, alpha=0.95)
        axes[i].axis("off")
        # axes[i].set_title(f"Time {frames[i]} Seconds", pad=20)

        # For each bounding box in this image:
        for j, box in enumerate(boxes_list[i]):
            pts = np.array(box).reshape(4, 2)
            # Draw the colored polygon (edge color remains 'red' here; adjust if needed).
            color_num = net_corrs[i][j]
            color = cmap(color_num % cmap.N + 1)
            face_color = (color[0], color[1], color[2], 0.3)
            poly = patches.Polygon(pts, closed=True, edgecolor=color, facecolor=face_color, linewidth=2)
            axes[i].add_patch(poly)
            # Compute the centroid of the box for labeling.
            # centroid = np.mean(pts, axis=0)
            # # Label the box using the net_corr value for this image and box index.
            # axes[i].text(centroid[0], centroid[1], str(net_corrs[i][j]),
            #              color='yellow', fontsize=12, ha='center', va='center')

    plt.tight_layout(pad=0.03, w_pad=0.03, h_pad=0.03, rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.close()

def rotate_image(img_file, rot_angle, save_dir, in_img=None):
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
    region_width = 1500 
    region_height = 1500 
    
    # Calculate center of the image
    center_x, center_y = w // 2, h // 2
    
    # Determine cropping boundaries for the center region.
    # Clamp boundaries if necessary (here we assume image dimensions are large enough).
    x1 = max(0, center_x - region_width // 2)
    x2 = min(w, center_x + region_width // 2)
    y1 = max(0, center_y - region_height // 2)
    y2 = min(h, center_y + region_height // 2)
    
    # Extract the center region.
    region = image[y1:y2, x1:x2]
    region_h, region_w = region.shape[:2]
    
    # Compute the rotation matrix.
    # OpenCV rotates counter-clockwise for positive angles; if you want clockwise, use -rot_angle
    M = cv2.getRotationMatrix2D((region_w / 2, region_h / 2), -rot_angle, 1.0)
    
    # Rotate the extracted region.
    rotated_region = cv2.warpAffine(region, M, (region_w, region_h))
    
    # Create a copy of the original image and replace the center region with the rotated region.
    new_image = image.copy()
    new_image[y1:y2, x1:x2] = rotated_region
    
    # Save the rotated image. The filename includes the rotation angle.
    save_file = os.path.join(save_dir, f"rotated_{rot_angle}_{img_file}")
    cv2.imwrite(save_file, new_image)
    print(f"Rotated image saved to {save_file}")

def translate_image(img_file, x, y, save_dir):
    image = cv2.imread(save_dir + img_file) 
    h, w = image.shape[:2]
    
    # Define the dimensions of the center region.
    region_width = 1500 
    region_height = 1500 
    
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
    save_file = save_dir + f"translated_{x}_{y}" + img_file
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

    # We’ll expand our output width so the entire sheared image shows.
    # For a horizontal shear:
    new_width = int(w + abs(shear_factor) * h)

    # Build the shear matrix.
    # Note the extra x-translation that shifts the image so it doesn’t go negative:
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

def generate_augmentations(img_dir):
    # CONFIG: TOGGLE FOR DIFFERENT AUGMENTATIONS
    rotation_angle_range = [-180, 180]
    translation_range = [-200, 200]
    shear_angle_range = [-10, 10]

    for img_file in os.listdir(img_dir):
        translation = np.random.randint(translation_range[0], translation_range[1], size=2)
        shear_angle = np.random.randint(shear_angle_range[0], shear_angle_range[1])
        rotation_angle1 = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
        rotation_angle2 = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
        translated_img = translate_image(img_file, translation[0], translation[1], img_dir)
        rotate_image(img_file, rotation_angle1, img_dir, in_img=translated_img)
        rotate_image(img_file, rotation_angle2, img_dir, in_img=translated_img)
        shear_image(img_file, shear_angle, img_dir)

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
        # img_1 = img_1[y1:y2, x1:x2]
        # img_2 = img_2[y1:y2, x1:x2] 

        # Save the images
        cv2.imwrite(f"pybullet_env/icp/img_0.jpg", img_1)
        cv2.imwrite(f"pybullet_env/icp/img_{i}.jpg", img_2)

        with torch.no_grad():
            img_1_path = f"pybullet_env/icp/img_0.jpg"
            img_2_path = f"pybullet_env/icp/img_{i}.jpg"
            results = model(source=[img_1_path, img_2_path],
                            conf=0.60,
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

if __name__ == "__main__":
    # generate_augmentations("pybullet_env/icp/whale_data/icp_experiments/shot1/")
    # pairwise_box_plots("pybullet_env/icp/whale_data/icp_experiments/shot1/")
    # find_num_boxes("pybullet_env/icp/whale_data/icp_experiments/shot3")

    # img_dir = "pybullet_env/icp/whale_data/icp_experiments/shot3"
    # dir_files = os.listdir(img_dir)
    # img_list = []
    # for _ in range(5):
    #     img_list.append(os.path.join(img_dir, random.choice(dir_files)))
    img_list = ["pybullet_env/icp/whale_data/icp_experiments/shot3/15_1688827660979_frame750.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_-119_19_1688827660979_frame950.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_-44_15_1688827660979_frame750.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/translated_-186_-14317_1688827660979_frame850.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_95_19_1688827660979_frame950.jpg"]
    calculate_box_plots(img_list)
