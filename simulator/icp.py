import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

def best_fit_transform(A, B):
    assert A.shape == B.shape

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
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, height_A, height_B, max_iters=20, tolerance=0.001):
    assert A.shape == B.shape, "A and B must have the same shape"

    # Calculate scaling factor
    scaling_factor = height_A / height_B
    scaling_factor = 1 # TODO: fixed for now for debugging purposes

    # Scale the points in A
    A_scaled = A * scaling_factor

    # Make points homogeneous, copy them to maintain the originals
    m = A.shape[1]
    src = np.ones((m+1, A_scaled.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A_scaled.T)
    dst[:m, :] = np.copy(B.T)

    prev_error = 0
    correspondence_indices = np.zeros(A.shape[0], dtype=int)

    for i in range(max_iters):
        # Find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # Update correspondence indices
        correspondence_indices = indices

        # Compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # Update the current source
        src = np.dot(T, src)

        # Check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Compute the final transformation
    T, R, t = best_fit_transform(A_scaled, src[:m, :].T)

    # Adjust the transformation matrix to account for the original scaling
    T[:m, :m] /= scaling_factor
    T[:m, m] /= scaling_factor

    # return T, distances, i, R, t, correspondence_indices
    return T, correspondence_indices

if __name__ == "__main__":
    # A = np.array([[1, 0], [1, 1], [4, 3], [-9, 1]])
    # theta = np.pi / 4
    # translation = np.array([-3, 2])
    # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # B = np.dot(A, rot_matrix.T) + translation
    A = np.array([[221.69948187, 15.61658031], [254.625, 17], [226.35869565, 29.17391304], [202.30508475, 42.76271186], [248.3, 43.02666667]])
    B = np.array([[191.51336898, 15.63076932], [234.93846154, 15.63076923], [196.45505618, 29.14044944], [172.02352941, 42.87058824], [220.37569061, 42.89502762]])
    # Heights at which A and B are viewed
    height_A = 10  
    height_B = 10 

    # Run ICP with height scaling
    T, correspondence_indices = icp(A, B, height_A, height_B)

    print(T)

    print("\nCorrespondence indices:", correspondence_indices)