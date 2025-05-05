import torch
from datetime import datetime
from ultralytics import YOLO
from icp import rot_icp 
from whale_icp_test import plot_boxed_images, rotate_image, translate_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def warp_affine_pixel(x, y, theta):
#     """
#     Given an (x, y) pixel in the full image coordinate space,
#     this applies the same transform as cv2.warpAffine in rotate_image().
#     """
#     # Translate to the cropped region’s local coordinates
#     M = cv2.getRotationMatrix2D((1920 / 2, 1920 / 2), -theta, 1.0)

#     # Apply OpenCV's warpAffine transform
#     # [x']   [ M[0,0]  M[0,1]  M[0,2] ] [local_x]
#     # [y'] = [ M[1,0]  M[1,1]  M[1,2] ] [local_y]
#     new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
#     new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]

#     # Translate back to full image coordinates
#     return new_x, new_y

def get_whale_obb_diagram(img_file):
    # Load image using cv2
    img = cv2.imread(img_file)
    
    # Load the YOLO model
    model_file = "pybullet_env/icp/whale_data/last.pt"
    model = YOLO(model_file)
    
    # Run the model on the input image
    results = model(source=[img_file],
                    conf=0.30,
                    imgsz=640,
                    classes=[0],
                    device="cpu")
                    
    # Extract oriented bounding boxes and confidence scores
    boxes = results[0].obb.xyxyxyxy.numpy()
    confs = results[0].obb.conf.cpu().numpy()
    
    # Plot the image and overlay the boxes and confidence scores with arrows pointing to their box
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Whale Boxes With Confidence Scores")  # New title added here
    ax = plt.gca()
    
    for box, conf in zip(boxes, confs):
        # Reshape flattened 8 values into 4 (x,y) coordinates
        coords = box.reshape(4, 2)
        poly = patches.Polygon(coords, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(poly)
        
        # Compute placement for confidence text: top-left corner of the box with small offset
        x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
        # Compute the centroid of the box for the arrow target
        centroid = np.mean(coords, axis=0)
        
        # Annotate with confidence score and an arrow pointing to the box centroid
        ax.annotate(f"{conf:.2f}",
                    xy=(centroid[0], centroid[1]),
                    xytext=(x_min, y_min - 5),
                    fontsize=8,
                    color='yellow',
                    bbox=dict(facecolor='black', edgecolor='none', pad=1),
                    arrowprops=dict(arrowstyle="->", color="yellow"))
    
    plt.axis('off')
    save_path = "pybullet_env/icp/whale_obb_diagram.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Whale OBB diagram saved to {save_path}")

def calculate_icp_plots(img0_file, img0_rot=0, img1_rot=0, img1_translation=None, padding=0):
    if not os.path.exists(f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}"):
        os.makedirs(f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}")

    model_file = "pybullet_env/icp/whale_data/last.pt"
    model = YOLO(model_file)
    img_0 = cv2.imread(img0_file)
    # Crop pixels from each border such that we get rid of weird border outline
    h, w = img_0.shape[:2]
    crop_size = 50
    img_0 = img_0[crop_size:h, :w-crop_size]
    # crop img_0 to get rid of border regions

    # rotate_image("", img0_rot, save_dir="pybullet_env/icp", in_img=img_0, custom_save_file="pybullet_env/icp/img_0.jpg", save=True)
    cv2.imwrite(f"pybullet_env/icp/img_0.jpg", img_0)

    # generate image 2 from augmentations
    base_rot = img1_rot 
    if img1_translation is None:
        init_dx, init_dy = 0, 0
    else:
        init_dx, init_dy = img1_translation
    img_1 = translate_image(img0_file, init_dx, init_dy, save_dir="", custom_save_file="pybullet_env/icp/img_1.jpg", vary_height=True, padding=padding)
    rotate_image("", base_rot, save_dir="pybullet_env/icp", in_img=img_1, custom_save_file="pybullet_env/icp/img_1.jpg", save=True)

    with torch.no_grad():
        img_0_path = f"pybullet_env/icp/img_0.jpg"
        img_1_path = f"pybullet_env/icp/img_1.jpg"
        results = model(source=[img_0_path, img_1_path],
                        conf=0.30,
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

    _, _, _, transforms = rot_icp(vid2_boxes, vid1_boxes, N=10, use_point=False, record=True)
    for t in transforms:
        rot_theta = np.rad2deg(t[0])
        incr_center = t[1]
        print(f"rot_theta: {rot_theta}")
        cum_transform_angle = 0
        cum_dx = init_dx
        cum_dy = init_dy
        i = 0

        # initial transform plot
        print(f"iter: {i}, rot: {base_rot + rot_theta + cum_transform_angle}, dx: {cum_dx}, dy: {cum_dy}")
        translation = translate_image("", 0, 0, save_dir="", in_img=img_0, padding=padding, vary_height=True)
        rot_image = rotate_image("", base_rot + rot_theta, save_dir="", in_img=translation, custom_save_file="pybullet_env/icp/pred_final.jpg", save=True)
        results = model(source=[rot_image],
                        conf=0.30,
                        imgsz=640,
                        classes=[0],
                        device=device,
                        verbose=False,
                    )
        final_boxes = results[0].obb.xyxyxyxy.numpy()
        plot_boxed_images(["pybullet_env/icp/img_0.jpg", "pybullet_env/icp/pred_final.jpg"], 
                            [vid1_boxes, final_boxes], 
                            [[i for i in range(len(vid1_boxes))], [i for i in range(len(vid1_boxes))]], 
                            save_path=f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}/init_rotation{rot_theta}_iter{i}.pdf",
                            plot_title=f"ICP Rotation {rot_theta} Iteration {i}")
        i += 1

        for T, corr, vid2_iter_boxes in zip(t[2], t[3], t[4]):
            transform_angle = np.rad2deg(np.arccos(min(1, T[0, 0])))
            cum_transform_angle += transform_angle
            # incr_center = warp_affine_pixel(incr_center[0], incr_center[1], transform_angle)    
            # vid2_iter_boxes += incr_center
            dx, dy = T[0, 2], T[1, 2]
            cum_dx += dx
            cum_dy += dy
            print(f"iter: {i}, rot: {base_rot + rot_theta + cum_transform_angle}, dx: {cum_dx}, dy: {cum_dy}")
            translation = translate_image("", 0, 0, save_dir="", in_img=img_0, padding=padding, vary_height=True) 
            rot_image = rotate_image("", base_rot + rot_theta + cum_transform_angle, save_dir="", in_img=translation, custom_save_file="pybullet_env/icp/pred_final.jpg", save=True) 

            # run model on the final transformed image
            results = model(source=[rot_image],
                            conf=0.30,
                            imgsz=640,
                            classes=[0],
                            device=device,
                            verbose=False,
                        )
            final_boxes = results[0].obb.xyxyxyxy.numpy()
            if final_boxes.shape[0] != num_boxes1: 
                continue
            _, corr, _ = rot_icp(final_boxes, vid1_boxes, N=10, use_point=False, record=False) 
            # final_center = np.mean(final_boxes.reshape(-1, 2), axis=0)
            # vid2_center = np.mean(vid2_iter_boxes.reshape(-1, 2), axis=0)
            # incr_center = final_center - vid2_center
            # vid2_iter_boxes += incr_center
            plot_boxed_images(["pybullet_env/icp/img_0.jpg", "pybullet_env/icp/pred_final.jpg"], 
                              [vid1_boxes, final_boxes], 
                              [[j for j in range(len(vid1_boxes))], corr], 
                              save_path=f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}/init_rotation{rot_theta}_iter{i}.pdf",
                              plot_title=f"ICP Rotation {rot_theta} Iteration {i}")
            i += 1
    
if __name__ == "__main__":
    print("STARTING PROGRAM")
    rotations = [0, -119, -44]
    # transformations = [[0, None], [-119, None], [-44, None], [0, (-186, -143)], [95, None]]
    img_1_rotation = 80 
    img_list = ["pybullet_env/icp/whale_data/final_model_eval/7_whale_3.png"]
    calculate_icp_plots(img_list[0], img1_rot=img_1_rotation, padding=0)
    # get_whale_obb_diagram(img_list[0])