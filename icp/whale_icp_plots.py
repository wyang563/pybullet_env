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

def calculate_icp_plots(img0_file, img0_rot=0, img1_rot=0, img1_translation=None):
    if not os.path.exists(f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}"):
        os.makedirs(f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}")

    model_file = "pybullet_env/icp/whale_data/last.pt"
    model = YOLO(model_file)
    img_0 = cv2.imread(img0_file)
    rotate_image("", img0_rot, save_dir="pybullet_env/icp", in_img=img_0, custom_save_file="pybullet_env/icp/img_0.jpg")
    # cv2.imwrite(f"pybullet_env/icp/img_0.jpg", img_0)

    # generate image 2 from augmentations
    base_rot = img1_rot 
    if img1_translation is None:
        init_dx, init_dy = 0, 0
    else:
        init_dx, init_dy = img1_translation
    img_1 = translate_image(img0_file, init_dx, init_dy, save_dir="", custom_save_file="pybullet_env/icp/img_1.jpg")
    rotate_image("", base_rot, save_dir="pybullet_env/icp", in_img=img_1, custom_save_file="pybullet_env/icp/img_1.jpg")

    with torch.no_grad():
        img_0_path = f"pybullet_env/icp/img_0.jpg"
        img_1_path = f"pybullet_env/icp/img_1.jpg"
        results = model(source=[img_0_path, img_1_path],
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

    _, _, _, transforms = rot_icp(vid2_boxes, vid1_boxes, use_point=False, record=True)
    for t in transforms:
        rot_theta = np.rad2deg(t[0])
        incr_center = t[1]
        print(f"rot_theta: {rot_theta}")
        cum_transform_angle = 0
        cum_dx = init_dx
        cum_dy = init_dy
        i = 0
        for T, corr, vid2_iter_boxes in zip(t[2], t[3], t[4]):
            transform_angle = np.rad2deg(np.arccos(min(1, T[0, 0])))
            cum_transform_angle += transform_angle
            # incr_center = warp_affine_pixel(incr_center[0], incr_center[1], transform_angle)    
            # vid2_iter_boxes += incr_center
            dx, dy = T[0, 2], T[1, 2]
            cum_dx += dx
            cum_dy += dy
            translation = translate_image(img0_file, cum_dx, cum_dy, save_dir="") 
            rotate_image("", base_rot + rot_theta + cum_transform_angle, save_dir="", in_img=translation, custom_save_file="pybullet_env/icp/pred_final.jpg") 

            # run model on the final transformed image
            results = model(source=["pybullet_env/icp/pred_final.jpg"],
                            conf=0.60,
                            imgsz=640,
                            classes=[0],
                            device="cpu",
                        )
            final_boxes = results[0].obb.xyxyxyxy.numpy()
            final_center = np.mean(final_boxes.reshape(-1, 2), axis=0)
            vid2_center = np.mean(vid2_iter_boxes.reshape(-1, 2), axis=0)
            incr_center = final_center - vid2_center
            vid2_iter_boxes += incr_center
            plot_boxed_images(["pybullet_env/icp/img_0.jpg", "pybullet_env/icp/pred_final.jpg"], 
                              [vid1_boxes, vid2_iter_boxes], 
                              [[i for i in range(len(vid1_boxes))], corr], 
                              save_path=f"pybullet_env/icp/icp_plots/img0_rot{img0_rot}_img1_rot{img1_rot}_img1_translation{img1_translation}/init_rotation{rot_theta}_iter{i}.pdf")
            i += 1
    
if __name__ == "__main__":
    rotations = [0, -119, -44]
    transformations = [[0, None], [-119, None], [-44, None], [0, (-186, -143)], [95, None]]
    img_list = ["pybullet_env/icp/whale_data/icp_experiments/shot3/15_1688827660979_frame750.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_-119_19_1688827660979_frame950.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_-44_15_1688827660979_frame750.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/translated_-186_-14317_1688827660979_frame850.jpg",
                "pybullet_env/icp/whale_data/icp_experiments/shot3/rotated_95_19_1688827660979_frame950.jpg"]
    for i in range(1, len(transformations)):
        t1 = transformations[i]
        t2 = transformations[(i + 1) % len(transformations)]
        calculate_icp_plots(img_list[0], img0_rot=t1[0], img1_rot=t2[0], img1_translation=t2[1]) 