import torch
from numpy.linalg import norm
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
from scipy.optimize import linear_sum_assignment


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
        plt.savefig(save_path, dpi=300)
    # plt.close()

def extract_boxes(img_0_path, img_1_path):
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
                        conf=0.50,
                        imgsz=640,
                        classes=[0],
                        device="cpu",
                    )
        vid1_boxes = results[0].obb.xyxyxyxy.numpy()
        vid2_boxes = results[1].obb.xyxyxyxy.numpy()
        num_boxes1 = vid1_boxes.shape[0]
        num_boxes2 = vid2_boxes.shape[0]
    
    if abs(num_boxes1 - num_boxes2) > 0:
        return None
    _, corr, _ = rot_icp(vid2_boxes, vid1_boxes, use_point=False)

    # plot image
    correlations = [corr] 
    correlations.insert(0, [i for i in range(len(correlations[0]))])
    boxes = [vid1_boxes, vid2_boxes]
    img_list = [img_0_path, img_1_path]
    plot_boxed_images(img_list, boxes, correlations, save_path=f"pybullet_env/icp/whale_data/box_plot.pdf")


def get_drone_to_whale_dists(boxes):
    '''
    Performs ICP to get consensus on whale positions between drones, and then calculates bounding boxes/distances
    between each whale and drone to then be used for goal assignment
    '''
    num_drones = len(boxes)
    all_whale_boxes = []
    correlations = []

    # setup debug directories in icp plots
    run_num = len(os.listdir("pybullet_env/icp/sim_data"))
    os.makedirs(f"pybullet_env/icp/sim_data/run_{run_num}")

    for d in range(1, len(boxes)):
        boxes1 = boxes[d]
        if d + 1 == len(boxes):
            boxes2 = boxes[0]
        else:
            boxes2 = boxes[d + 1]
        
        # center all boxes at origin
        centroid_boxes1 = np.mean(boxes1, axis=0) 
        centroid_boxes2 = np.mean(boxes2, axis=0)
        centered_boxes1 = boxes1 - centroid_boxes1
        centered_boxes2 = boxes2 - centroid_boxes2

        _, corr, _ = rot_icp(centered_boxes2, centered_boxes1, N = 50, use_point=False)                
        correlations.append(corr)

        # add centered boxes 1 to all whale boxes at the end
        all_whale_boxes.append(centered_boxes1)

    # calculate net correlations
    net_corrs = [] # net_corrs[i] is the mapping from point cloud i to point cloud 0
    for i in range(len(correlations)):
        composite = np.arange(len(correlations[i]))
        for j in range(i, -1, -1):
            new_composite = np.zeros(len(composite), dtype=int)
            for k in range(len(composite)):
                new_composite[correlations[j][k]] = composite[k]
            composite = new_composite.copy()
        net_corrs.append(composite)
    
    front = net_corrs.pop()
    net_corrs.insert(0, front)
    print(net_corrs)

    num_objects = len(net_corrs[0])

    # check the mapping is the identity at the end
    if net_corrs[0].tolist() != [i for i in range(num_objects)]:
        assert False, f"final net correlation was not the identity mapping: {net_corrs[0]}"
    
    # reshuffle all_whale_boxes
    for i in range(len(all_whale_boxes)):
        corr = net_corrs[i]
        all_whale_boxes[i] = [all_whale_boxes[i][corr[j]] for j in range(len(all_whale_boxes[i]))]
    
    cost_matrix = np.zeros(num_drones, num_objects)
    
    for i in range(num_drones):
        for j in range(num_objects):
            whale_center = np.mean(all_whale_boxes[i][j], axis=0)
            cost_matrix[i][j] = norm(whale_center)
    
    # do linear sum assignment on cost matrix to get drone assignments
    drone_assignments = linear_sum_assignment(cost_matrix)
    return all_whale_boxes, cost_matrix, drone_assignments, net_corrs
    

    


