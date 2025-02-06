import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm

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

def calc_all_box_distance(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape, "expected matrix of boxes for both inputs to have the same shape"
    pairwise_box_dists = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    pairwise_corr_indices = np.zeros((boxes1.shape[0], boxes2.shape[0], 4))
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            pairwise_corr_indices[i, j], pairwise_box_dists[i, j] = calc_pair_box_distance(boxes1[i], boxes2[j])
    box_corr_row_ind, box_corr_col_ind = linear_sum_assignment(pairwise_box_dists)
    flat_corr_indices = np.zeros((boxes1.shape[0] * 4), dtype=int)
    box_assign_dist = 0
    for row_ind, col_ind in zip(box_corr_row_ind, box_corr_col_ind):
        box_assign_dist += pairwise_box_dists[row_ind, col_ind]
        for i, ind in enumerate(pairwise_corr_indices[row_ind, col_ind]):
            flat_corr_indices[row_ind * 4 + i] = 4 * col_ind + ind
    return box_assign_dist, flat_corr_indices.ravel(), box_corr_col_ind

def icp(A, B, max_iters=20, tolerance=0.0001):
    assert A.shape == B.shape, "A and B must have the same shape"

    # Make points homogeneous, copy them to maintain the originals
    prev_error = 0
    correspondence_indices = np.zeros(A.shape[0], dtype=int)
    original_A = np.copy(A.reshape(-1, 2))

    for i in range(max_iters):
        # convert coordinates to homogenous flattened list of coordinates
        A_flattened = A.reshape(-1, 2)
        B_flattened = B.reshape(-1, 2)
        m = A_flattened.shape[1]
        src = np.ones((m+1, A_flattened.shape[0])) 
        dst = np.ones((m+1, B_flattened.shape[0]))
        src[:m, :] = np.copy(A_flattened.T)
        dst[:m, :] = np.copy(B_flattened.T)

        # Find the nearest neighbors between the current source and destination points
        distances, indices, box_assign_inds = calc_all_box_distance(A, B)
        correspondence_indices = box_assign_inds

        # Compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        
        # Update the current source and update A (N, 4, 2) matrix
        src = np.dot(T, src)
        A = np.zeros((A.shape[0], 4, 2))
        A = np.copy(src[:src.shape[0] - 1, :].T.reshape(-1, 4, 2))

        # Check error
        if np.abs(prev_error - distances) < tolerance:
            break
        prev_error = distances

    # Compute the final transformation
    T, _, _ = best_fit_transform(original_A, src[:m, :].T)
    return T, correspondence_indices

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