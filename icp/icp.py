import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

def plot_point_clouds(index, points1, points2, run_number):
    centroids1 = np.mean(points1, axis=1)
    centroids2 = np.mean(points2, axis=1)
    # points1_flat = points1.reshape(-1, 2)
    # points2_flat = points2.reshape(-1, 2)
    plt.figure(figsize=(6, 6))
    # plt.scatter(points1_flat[:, 0], points1_flat[:, 1], color='orange', label='Points 1')
    # plt.scatter(points2_flat[:, 0], points2_flat[:, 1], color='blue', label='Points 2')
    plt.scatter(centroids1[:, 0], centroids1[:, 1], color='red', label='Centroids 1')
    plt.scatter(centroids2[:, 0], centroids2[:, 1], color='green', label='Centroids 2')
    plt.title(f"Plot Point Clouds visualized {index}")
    plt.savefig(f"pybullet_env/simulator/icp_plots/point_clouds_{run_number}_{index}.png")
    plt.axis('equal')
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

def calc_pair_box_distance(box1, box2):
    '''
    given two rectangular bounding boxes, calculate the minimum bijective pairwise 
    distance between vertices of the two boxes.
    '''
    assert box1.shape == (4, 2) and box2.shape == (4, 2), "expected box inputs to have shape 4x2, instead found shapes {} and {}".format(box1.shape, box2.shape)
    pairwise_distances = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            pairwise_distances[i, j] = norm(box1[i] - box2[j])
    row_ind, col_ind = linear_sum_assignment(pairwise_distances)
    return col_ind, np.sum(pairwise_distances[row_ind, col_ind])   

def calc_all_box_distance(boxes1, boxes2, use_centers=False):
    assert boxes1.shape == boxes2.shape, "expected matrix of boxes for both inputs to have the same shape"
    num_boxes = boxes1.shape[0]
    pairwise_box_dists = np.zeros((num_boxes, num_boxes))
    pairwise_corr_indices = np.zeros((num_boxes, num_boxes, 4), dtype=int)
    pairwise_corr_indices[...] = np.arange(4)
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            if use_centers:
                center1 = np.mean(boxes1[i], axis=0)
                center2 = np.mean(boxes2[j], axis=0)
                pairwise_box_dists[i, j] = norm(center1 - center2)
            else:
                pairwise_corr_indices[i, j], pairwise_box_dists[i, j] = calc_pair_box_distance(boxes1[i], boxes2[j])

    #nearest_indices = np.argmin(pairwise_box_dists, axis=1)

    """
    flat_corr_indices = np.zeros((num_boxes * 4), dtype=int)
    box_assign_dist = 0
    for i in range(num_boxes):
        j = nearest_indices[i]
        box_assign_dist += pairwise_box_dists[i, j]
        corner_map = pairwise_corr_indices[i, j]
        for c, corner_j in enumerate(corner_map):
            flat_corr_indices[i * 4 + c] = 4 * j + corner_j
    """
    #print(nearest_indices.shape)
    #return box_assign_dist, flat_corr_indices.ravel(), nearest_indices 
    #print(flat_corr_indices.shape)

    box_corr_row_ind, box_corr_col_ind = linear_sum_assignment(pairwise_box_dists)
    box_assign_dist = 0
    box_assign_dist = pairwise_box_dists[box_corr_row_ind, box_corr_col_ind]
    flat_corr_indices = np.zeros((num_boxes * 4), dtype=int)
    box_assign_dist = 0
    for i in range(num_boxes):
        j = box_corr_row_ind[i]
        #box_assign_dist += pairwise_box_dists[i, j]
        corner_map = pairwise_corr_indices[i, j]
        for c, corner_j in enumerate(corner_map):
            flat_corr_indices[i * 4 + c] = 4 * j + corner_j
    return box_assign_dist, flat_corr_indices.ravel(), box_corr_row_ind

def icp(A, B, max_iters=20, tolerance=0.0001, use_centers=False, outlier_sigma=2, run_number=0):
    assert A.shape == B.shape, "A and B must have the same shape"

    # Make points homogeneous, copy them to maintain the originals
    prev_error = 0
    correspondence_indices = np.zeros(A.shape[0], dtype=int)
    original_A = np.copy(A.reshape(-1, 2))

    for i in range(max_iters):
        # convert coordinates to homogenous flattened list of coordinates
        A_flattened = A.reshape(-1, 2)
        B_flattened = B.reshape(-1, 2)
            
        # Find the nearest neighbors between the current source and destination points
        distances, indices, box_assign_inds = calc_all_box_distance(A, B, use_centers)
        pair_dists = np.array([np.linalg.norm(A_flattened[k] - B_flattened[idx]) 
                               for k, idx in enumerate(indices)])
        mean_dist = np.mean(pair_dists)
        std_dist = np.std(pair_dists)
        threshold = mean_dist + outlier_sigma * std_dist
        inlier_mask = pair_dists < threshold
        inlier_A = A_flattened[inlier_mask]
        inlier_B = B_flattened[indices[inlier_mask]]
        m = A_flattened.shape[1] 

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
        A = np.zeros((A.shape[0], 4, 2))
        A = np.copy(org_src[:org_src.shape[0] - 1, :].T.reshape(-1, 4, 2))
        # plot_point_clouds(i, A, B, run_number)

        # Check error
        if np.abs(prev_error - distances) < tolerance:
            break
        prev_error = distances

    # Compute the final transformation
    T, _, _ = best_fit_transform(original_A, org_src[:m, :].T)

    # compute the final correspondence indices, these are bijective

    return T, correspondence_indices, prev_error 

def rot_icp(A, B, use_centers=False):
    '''
    Rotatet A by many random increments, and return the correspondence indices with the lowest error
    '''
    N = 100
    ll = 0
    for _ in range(10):
        lowest_error = float('inf')
        low_correspondence = None
        low_transform = None
        for _ in range(N):
            rot_theta = np.deg2rad(random.uniform(0, 360))
            R = np.array([[np.cos(rot_theta), -np.sin(rot_theta)], [np.sin(rot_theta), np.cos(rot_theta)]])
            transformed_points = []
            for box in A:
                transformed_box = []
                for point in box:
                    point_array = np.array(point)
                    #print(R.shape, point_array.shape)
                    transformed_point = np.dot(R, point_array)
                    transformed_box.append(transformed_point.tolist())
                transformed_points.append(transformed_box)

            T, corr_indices, error = icp(np.array(transformed_points), B, use_centers=use_centers, run_number=rot_theta)        
            if error < lowest_error:
                lowest_error = error 
                low_correspondence = corr_indices
                low_transform = T

        if len(set(low_correspondence)) == len(low_correspondence):
            return low_transform, low_correspondence, lowest_error
        else:
            ll+=1
            print("look",ll)

    return low_transform, low_correspondence, lowest_error 

if __name__ == "__main__":
    # A = np.array([[1, 0], [1, 1], [4, 3], [-9, 1]])
    # theta = np.pi / 4
    # translation = np.array([-3, 2])
    # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # B = np.dot(A, rot_matrix.T) + translation
    A = np.array([[[208,  13],
        [227,  13],
        [227,  23],
        [208,  23]],

       [[250,  14],
        [255,  14],
        [255,  23],
        [250,  23]],

       [[212,  26],
        [232,  26],
        [232,  37],
        [212,  37]],

       [[189,  40],
        [208,  40],
        [208,  50],
        [189,  50]],

       [[236,  40],
        [255,  40],
        [255,  50],
        [236,  50]]])

    B = np.array([[[175,  22],
        [175,  11],
        [194,  11],
        [194,  22]],

       [[218,  11],
        [238,  11],
        [238,  22],
        [218,  22]],

       [[179,  25],
        [198,  25],
        [198,  36],
        [179,  36]],

       [[155,  39],
        [174,  39],
        [174,  49],
        [155,  49]],

       [[203,  39],
        [223,  39],
        [223,  49],
        [203,  49]]]) 
    T, corr_indices = icp(A, B) 
    print(T, corr_indices)