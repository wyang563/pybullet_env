import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm, eigh 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.neighbors import NearestNeighbors
import json
import os

def plot_point_clouds(index, points1, points2, indices, run_number):
    # centroids1 = np.mean(points1, axis=1)
    # centroids2 = np.mean(points2, axis=1)
    points1_flat = points1.reshape(-1, 2)
    points2_flat = points2.reshape(-1, 2)
    plt.figure(figsize=(6, 6))
    plt.scatter(points1_flat[:, 0], points1_flat[:, 1], color='orange', label='Points 1')
    plt.scatter(points2_flat[:, 0], points2_flat[:, 1], color='blue', label='Points 2')

    # Label each point in points1_flat
    for i, (x, y) in enumerate(points1_flat):
        plt.text(x, y, str(indices[i]), color='orange', fontsize=8, ha='center', va='bottom')
    
    # Label each point in points2_flat
    for i, (x, y) in enumerate(points2_flat):
        plt.text(x, y, str(i), color='blue', fontsize=8, ha='center', va='bottom')
    # plt.scatter(centroids1[:, 0], centroids1[:, 1], color='red', label='Centroids 1')
    # plt.scatter(centroids2[:, 0], centroids2[:, 1], color='green', label='Centroids 2')
    plt.title(f"Plot Point Clouds visualized: {index}")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f"pybullet_env/icp/icp_plots/point_clouds_{run_number}_{index}.png")
    plt.close()

def plot_rectangles(index, points1, points2, indices, run_number):
    """
    index: identifier for this plot
    points1, points2: np.ndarray of shape (N,4,2) listing rectangle corner coords
    indices: length-N array mapping each rectangle in points1 to a rectangle in points2
    run_number: used in output filename
    """
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # draw rectangles
    for i, rect in enumerate(points1):
        poly = patches.Polygon(rect, closed=True, edgecolor='orange', fill=False, linewidth=2)
        ax.add_patch(poly)
        # centroid
        c1 = rect.mean(axis=0)
        ax.text(c1[0], c1[1], str(i), color='orange', fontsize=8, ha='center', va='center')

    for j, rect in enumerate(points2):
        poly = patches.Polygon(rect, closed=True, edgecolor='blue', fill=False, linewidth=2)
        ax.add_patch(poly)
        c2 = rect.mean(axis=0)
        ax.text(c2[0], c2[1], str(j), color='blue', fontsize=8, ha='center', va='center')

    # draw correspondences
    for i, j in enumerate(indices):
        c1 = points1[i].mean(axis=0)
        c2 = points2[j].mean(axis=0)
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color='gray', linestyle='--', linewidth=1)

    plt.title(f"Rectangles correspondence: {index}")
    out_dir = "pybullet_env/icp/icp_plots"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/rectangles_{run_number}_{index}.png")
    plt.close()

def best_fit_transform(A, B):
    assert A.shape == B.shape, "found shapes A: {} and B: {}".format(A.shape, B.shape)

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape
    centeroid_src = np.mean(src, axis=0)
    centeroid_dst = np.mean(dst, axis=0)
    src_centered = src - centeroid_src
    dst_centered = dst - centeroid_dst
    costs = np.zeros((src.shape[0], dst.shape[0]))
    for i in range(src.shape[0]):
        for j in range(dst.shape[0]):
            costs[i, j] = norm(src_centered[i] - dst_centered[j])
    _, indices = linear_sum_assignment(costs)
    distances = np.array([costs[i, idx] for i, idx in enumerate(indices)])
    return distances, indices

def calc_pair_box_distance(box1, box2, box_size):
    '''
    given two rectangular bounding boxes, calculate the minimum bijective pairwise 
    distance between vertices of the two boxes.
    '''
    assert box1.shape == (box_size, 2) and box2.shape == (box_size, 2), "expected box inputs to have shape 4x2, instead found shapes {} and {}".format(box1.shape, box2.shape)
    box1_center = np.mean(box1, axis=0)
    box2_center = np.mean(box2, axis=0)
    box1_centered = box1 - box1_center
    box2_centered = box2 - box2_center
    pairwise_distances = np.zeros((box_size, box_size))
    for i in range(box_size):
        for j in range(box_size):
            pairwise_distances[i, j] = norm(box1_centered[i] - box2_centered[j])
    
    # assign indices according to closest vertex
    _, box_nearest_indices = linear_sum_assignment(pairwise_distances)

    # calculate distance between centers of boxes
    return box_nearest_indices, norm(box1_center - box2_center) 

def calc_all_box_distance(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape, "expected matrix of boxes for both inputs to have the same shape"
    num_boxes = boxes1.shape[0]
    box_size = boxes1.shape[1]
    pairwise_box_dists = np.zeros((num_boxes, num_boxes))
    pairwise_corr_indices = np.zeros((num_boxes, num_boxes, box_size), dtype=int)
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            pairwise_corr_indices[i, j], pairwise_box_dists[i, j] = calc_pair_box_distance(boxes1[i], boxes2[j], box_size)

    # nearest_indices = np.argmin(pairwise_box_dists, axis=1)
    _, nearest_indices = linear_sum_assignment(pairwise_box_dists)
    flat_corr_indices = np.zeros((num_boxes * box_size), dtype=int)
    box_assign_dist = 0
    for i in range(num_boxes):
        j = nearest_indices[i]
        box_assign_dist += pairwise_box_dists[i, j]
        corner_map = pairwise_corr_indices[i, j]
        for c, corner_j in enumerate(corner_map):
            flat_corr_indices[i * box_size + c] = box_size * j + corner_j
    return box_assign_dist, flat_corr_indices.ravel(), nearest_indices 

def icp(A, B, max_iters=20, tolerance=0.0001, outlier_sigma=2, record_transforms=False):
    assert A.shape == B.shape, "A and B must have the same shape"

    # Make points homogeneous, copy them to maintain the originals
    prev_error = float('inf')
    correspondence_indices = np.zeros(A.shape[0], dtype=int)
    original_A = np.copy(A.reshape(-1, 2))
    box_size = A.shape[1]
    transforms = []
    iter_correlations = []
    point_clouds = []

    for i in range(max_iters):
        A_flattened = A.reshape(-1, 2)
        B_flattened = B.reshape(-1, 2)

        # Find the nearest neighbors between the current source and destination points
        distances, indices, box_assign_inds = calc_all_box_distance(A, B)
        pair_dists = np.array([np.linalg.norm(A_flattened[k] - B_flattened[idx]) 
                                for k, idx in enumerate(indices)])
        mean_dist = np.mean(pair_dists)
        std_dist = np.std(pair_dists)
        threshold = mean_dist + outlier_sigma * std_dist
        inlier_mask = pair_dists < threshold
        inlier_A = A_flattened[inlier_mask]
        inlier_B = B_flattened[indices[inlier_mask]]
        m = A_flattened.shape[1] 

        # plot_point_clouds(i, A, B, indices, run_number)

        src = np.ones((m+1, inlier_A.shape[0])) 
        dst = np.ones((m+1, inlier_B.shape[0]))
        src[:m, :] = np.copy(inlier_A.T)
        dst[:m, :] = np.copy(inlier_B.T)

        correspondence_indices = box_assign_inds

        # Compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, :].T)
        
        # Update the current source and update A (N, 4, 2) matrix
        org_src = np.ones((m+1, A_flattened.shape[0]))
        org_src[:m, :] = np.copy(A_flattened.T) 
        org_src = np.dot(T, org_src)
        A = np.zeros((A.shape[0], box_size, 2))
        A = np.copy(org_src[:org_src.shape[0] - 1, :].T.reshape(-1, box_size, 2))
        # add data 
        if record_transforms:
            transforms.append(T.copy())
            iter_correlations.append(correspondence_indices.copy())
            point_clouds.append(A.copy())

        # Check error
        if np.abs(prev_error - distances) < tolerance:
            break
        prev_error = distances

    # Compute the final transformation
    T, _, _ = best_fit_transform(original_A, org_src[:m, :].T)
    if record_transforms:
        return T, correspondence_indices, prev_error, transforms, iter_correlations, point_clouds
    return T, correspondence_indices, prev_error 

def point_icp(A, B, max_iterations=20, tolerance=0.0001, run_number=0):
    # A_centers = np.mean(A, axis=1)
    # B_centers = np.mean(B, axis=1)
    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    prev_error = float('inf')

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # plot_point_clouds(i, src[:m, :].T, B, indices, run_number)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    return T, indices, mean_error

def rot_icp(A, B, N=30, use_point=False, record=False):
    '''
    Rotatet A by many random increments, and return the correspondence indices with the lowest error
    '''
    # make A, B centered around origin
    A_global_center = np.mean(A.reshape(-1, 2), axis=0)
    B_global_center = np.mean(B.reshape(-1, 2), axis=0)
    A = A - A_global_center
    B = B - B_global_center

    lowest_error = float('inf')
    low_correspondence = None
    low_transform = None
    recorded_clouds = []
    for i in range(N):
        rot_theta = np.deg2rad(360 / N * i)
        R = np.array([[np.cos(rot_theta), -np.sin(rot_theta)], [np.sin(rot_theta), np.cos(rot_theta)]])
        transformed_points = []
        if not use_point:
            for box in A:
                transformed_box = []
                for point in box:
                    point_array = np.array(point)
                    # Ensure point_array is (2,) or (1,2) for R (2,2) dot point_array.T or point_array
                    transformed_point = np.dot(R, point_array.T).T 
                    transformed_box.append(transformed_point.tolist())
                transformed_points.append(transformed_box)

            if record:
                T, corr_indices, error, transforms, iter_correlations, point_clouds = icp(np.array(transformed_points), B, record_transforms=True)        
            else:
                T, corr_indices, error = icp(np.array(transformed_points), B, record_transforms=False)
        else:
            # A is likely (num_points, 2) or needs to be reshaped if it's (num_boxes, 4, 2)
            # If A is (num_boxes, 4, 2) and we want to treat all corners as points:
            A_reshaped = A.reshape(-1, 2) # Reshape to (num_boxes * 4, 2)
            # add noise to A_reshaped, making the noise standard deviation proportional to the coordinate values
            noise_proportion_factor = 0.1 # Adjust this factor as needed
            for point_coords in A_reshaped: # point_coords will be (2,)
                transformed_point = np.dot(R, point_coords.T).T
                transformed_points.append(transformed_point.tolist())
            # transformed_points will be a list of lists, e.g., [[x1,y1], [x2,y2], ...]
            # point_icp expects an array of shape (N, 2)
            T, corr_indices, error = point_icp(np.array(transformed_points), B.reshape(-1,2))
        
        if error < lowest_error:
            lowest_error = error 
            low_correspondence = corr_indices
            low_transform = T
            if record:
                recorded_clouds.append((rot_theta, A_global_center, transforms, iter_correlations, point_clouds))
    if not record:
        return low_transform, low_correspondence, lowest_error 
    else:
        return low_transform, low_correspondence, lowest_error, recorded_clouds 

if __name__ == "__main__":
    # A = np.array([[1, 0], [1, 1], [4, 3], [-9, 1]])
    # theta = np.pi / 4
    # translation = np.array([-3, 2])
    # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # B = np.dot(A, rot_matrix.T) + translation
    points_file = "pybullet_env/icp/sim_data/run_47/points.json" 
    with open(points_file, "r") as f:
        data = json.load(f)
    A = np.array(data[1])
    B = np.array(data[2])
    T, corr_indices, error = rot_icp(A, B, use_point=False)



