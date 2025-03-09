import torch
from datetime import datetime
from ultralytics import YOLO
from icp import rot_icp 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_boxed_images(img1_path, boxes1, img2_path, boxes2, corr, save_path=None):
    """
    Reads in two images, draws the oriented bounding boxes from boxes1 and boxes2 onto img1 and img2 respectively,
    labels each box with its index, and displays the images side by side.
    
    Args:
        img1_path (str): Path to the first image.
        boxes1 (np.ndarray): Bounding boxes for the first image in xyxyxyxy format (each row contains eight numbers representing four (x,y) pairs).
        img2_path (str): Path to the second image.
        boxes2 (np.ndarray): Bounding boxes for the second image in xyxyxyxy format.
        save_path (str): Optional. File path to save the resulting plot.
    """
    # Read images and convert from BGR (cv2 default) to RGB
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first image
    ax1.imshow(image1)
    # For each bounding box in boxes1, create a polygon patch and label it
    for i, box in enumerate(boxes1):
        pts = np.array(box).reshape(4, 2)
        poly = patches.Polygon(pts, closed=True, edgecolor='red', facecolor='none', linewidth=2)
        ax1.add_patch(poly)
        # Compute the centroid of the box for labeling
        centroid = np.mean(pts, axis=0)
        ax1.text(centroid[0], centroid[1], str(i), color='yellow', fontsize=12, ha='center', va='center')
    ax1.set_title("Image 1")
    ax1.axis("off")

    # Plot the second image
    ax2.imshow(image2)
    for i, box in enumerate(boxes2):
        pts = np.array(box).reshape(4, 2)
        poly = patches.Polygon(pts, closed=True, edgecolor='red', facecolor='none', linewidth=2)
        ax2.add_patch(poly)
        # Compute the centroid for labeling the box index
        centroid = np.mean(pts, axis=0)
        ax2.text(centroid[0], centroid[1], str(corr[i]), color='yellow', fontsize=12, ha='center', va='center')
    ax2.set_title("Image 2")
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    if not os.path.exists("pybullet_env/icp/icp_plots"):
        os.makedirs("pybullet_env/icp/icp_plots")
    video_1_prefix = "pybullet_env/icp/whale_data/video_1/"
    video_2_prefix = "pybullet_env/icp/whale_data/video_2/"
    model = YOLO("pybullet_env/icp/whale_data/last.pt")
    run_name = f"yolorun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    index = 2000 
    img_1_path = video_1_prefix + f"{index}.jpg"
    img_1 = cv2.imread(img_1_path)
    height, width = img_1.shape[:2]

    # Crop the center 2000x2000 portion
    crop_size = 2300
    center_x, center_y = width // 2, height // 2
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    img_1 = img_1[y1:y2, x1:x2]

    # Rotate img_1 by 90 degrees to get img_2
    img_2 = cv2.rotate(img_1, cv2.ROTATE_90_CLOCKWISE)

    # Save the images
    cv2.imwrite("pybullet_env/icp/cropped_img_1.jpg", img_1)
    cv2.imwrite("pybullet_env/icp/rotated_img_2.jpg", img_2)

    with torch.no_grad():
        img_1_path = "pybullet_env/icp/cropped_img_1.jpg"
        img_2_path = "pybullet_env/icp/rotated_img_2.jpg"
        results = model(source=[img_1_path, img_2_path],
                        conf=0.45,
                        imgsz=640,
                        classes=[0],
                        device="cpu",
                        # project="/home/gridsan/wyang/super_urop_workspace/multienv_sim/pybullet_env/icp/logs",
                        # name=run_name,
                        # show=True,
                        # save_txt=True,
                        # show_boxes=True,
                        # show_labels=True,
                        # show_conf=True,
                        # save=True
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
        
        _, corr, _ = rot_icp(vid1_boxes, vid2_boxes)

    # plot results
    plot_boxed_images(img_1_path, vid1_boxes, img_2_path, vid2_boxes, corr, save_path="pybullet_env/icp/icp_plots/boxed_plot.png")