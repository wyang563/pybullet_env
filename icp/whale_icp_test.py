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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    n_images = len(img_list)
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
    
    # If only one image, make axes iterable
    if n_images == 1:
        axes = [axes]
    
    frames = [0, 2, 4, 6, 8]
    
    for i in range(n_images):
        # Read and convert the image from BGR to RGB
        image = cv2.imread(img_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(f"Time {frames[i]} Seconds", pad=20)
        # For each bounding box in this image
        for j, box in enumerate(boxes_list[i]):
            pts = np.array(box).reshape(4, 2)
            poly = patches.Polygon(pts, closed=True, edgecolor='red', facecolor='none', linewidth=2)
            axes[i].add_patch(poly)
            # Compute the centroid of the box for labeling
            centroid = np.mean(pts, axis=0)
            # Label the box using the net_corr value for this image and box index
            axes[i].text(centroid[0], centroid[1], str(net_corrs[i][j]),
                         color='yellow', fontsize=12, ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    region_width = 1920
    region_height = 1080
    
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
    region_width = 1920
    region_height = 1080
    
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
    # Read the image
    if in_img is None:
        image = cv2.imread(os.path.join(save_dir, img_file))
        if image is None:
            print(f"Error: Unable to read image {os.path.join(save_dir, img_file)}")
            return
    else:
        image = in_img 

    h, w = image.shape[:2]

    # Convert angle in degrees to a shear factor (tan of the angle)
    shear_factor = math.tan(math.radians(shear_angle))

    # Build the shear matrix.
    # This applies a horizontal shear; pixels get shifted in x by an amount 
    # proportional to their y-coordinate and the shear factor.
    M = np.float32([
        [1, shear_factor, 0],
        [0, 1,          0]
    ])

    # Warp (sheer) the entire image; we use the original image size for output.
    # Note: parts of the image may be clipped depending on the shear angle.
    sheered_image = cv2.warpAffine(image, M, (w, h))

    # Save the sheered image under a new name.
    out_filename = f"sheered_{shear_angle}_{img_file}"
    out_path = os.path.join(save_dir, out_filename)
    cv2.imwrite(out_path, sheered_image)
    print(f"Sheered image saved to {out_path}")

def generate_augmentations(img_dir):
    # CONFIG: TOGGLE FOR DIFFERENT AUGMENTATIONS
    rotation_angle_range = [-180, 180]
    translation_range = [-500, 500]
    shear_angle_range = [-25, 25]

    for img_file in os.listdir(img_dir):
        translation = np.random.randint(translation_range[0], translation_range[1], size=2)
        shear_angle = np.random.randint(shear_angle_range[0], shear_angle_range[1])
        rotation_angle = np.random.randint(rotation_angle_range[0], rotation_angle_range[1])
        
        translated_img = translate_image(img_file, translation[0], translation[1], img_dir)
        rotate_image(img_file, rotation_angle, img_dir, in_img=translated_img)
        shear_image(img_file, shear_angle, img_dir, in_img=translated_img)

def calculate_box_plots():
    if not os.path.exists("pybullet_env/icp/icp_plots"):
        os.makedirs("pybullet_env/icp/icp_plots")
    
    is_yolo_dataset = True
    
    # CONFIG: TOGGLE FOR DIFFERENT VIDEOS
    # config 1: video 2, start frame 0
    # config 2: video 3, start frame 0

    if not is_yolo_dataset:
        # video_num = 2
        # start_frame = 200 
        # stagger = 150 

        video_num = 3
        start_frame = 1000
        stagger = 50

        video_dir = f"pybullet_env/icp/whale_data/video_{video_num}"
    else:
        video_dir = "pybullet_env/icp/whale_data/yolo_dataset/images/val/"
        video_num = 1688827660979
        start_frame = 4850 
        start_ind = 97 
        stagger = 50 # this is the gap we want between frames in our box analysis
        gap = 50 # this is the gap between consecutive shots in the same video

    model_file = "pybullet_env/icp/whale_data/last.pt"

    model = YOLO(model_file)
    run_name = f"yolorun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    correlations = []
    img_list = []
    boxes = []
    for i, diff in enumerate([stagger * (i + 1) for i in range(4)]):
        frame_2 = start_frame + diff
        
        if not is_yolo_dataset:
            img_1_path = video_dir + f"{start_frame}.jpg"
            img_1 = cv2.imread(img_1_path)
            height, width = img_1.shape[:2]

            img_2_path = video_dir + f"{frame_2}.jpg"
            img_2 = cv2.imread(img_2_path)
        else:
            img_1_path = video_dir + f"{start_ind}_{video_num}_frame{start_frame}.jpg"
            img_1 = cv2.imread(img_1_path)
            height, width = img_1.shape[:2]

            img_2_path = video_dir + f"{start_ind + (i + 1) * stagger // gap}_{video_num}_frame{frame_2}.jpg"
            img_2 = cv2.imread(img_2_path)

        # Crop the center 2000x2000 portion
        crop_size = 2150
        center_x, center_y = width // 2, height // 2
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(width, center_x + crop_size // 2)
        y2 = min(height, center_y + crop_size // 2)
        img_1 = img_1[y1:y2, x1:x2]
        img_2 = img_2[y1:y2, x1:x2] 

        # Save the images
        cv2.imwrite(f"pybullet_env/icp/img_{start_frame}.jpg", img_1)
        cv2.imwrite(f"pybullet_env/icp/img_{frame_2}.jpg", img_2)

        with torch.no_grad():
            img_1_path = f"pybullet_env/icp/img_{start_frame}.jpg"
            img_2_path = f"pybullet_env/icp/img_{frame_2}.jpg"
            results = model(source=[img_1_path, img_2_path],
                            conf=0.6,
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
            if num_boxes1 > num_boxes2:
                vid1_boxes = vid1_boxes[:-diff]
            elif num_boxes2 > num_boxes1:
                vid2_boxes = vid2_boxes[:-diff]

        # get centers of boxes
        # vid1_centers = np.mean(vid1_boxes, axis=1)
        # vid2_centers = np.mean(vid2_boxes, axis=1)

        _, corr, _ = rot_icp(vid2_boxes, vid1_boxes, use_point=False)
        if i == 0:
            img_list.append(img_1_path)
            boxes.append(vid1_boxes)
        img_list.append(img_2_path)
        boxes.append(vid2_boxes)
        correlations.append(corr.tolist())

    # plot results
    correlations.insert(0, [i for i in range(len(correlations[0]))])
    print(correlations)
    print(img_list)
    plot_boxed_images(img_list, boxes, correlations, save_path="pybullet_env/icp/boxed_plot.png")

if __name__ == "__main__":
    img_dir = "pybullet_env/icp/whale_data/icp_experiments/shot1/"
    img_file = "503_1688841618482_frame420.jpg"
    # rotate_image(img_file, 180, save_dir=img_dir)
    # translate_image(img_file, 100, 100, save_dir=img_dir)
    # shear_image(img_file, 10, save_dir=img_dir)
    generate_augmentations(img_dir)