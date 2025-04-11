from icp.icp import rot_icp, icp
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import sys

def generate_random_rectangle(ax_aligned=True, grid_size=100):
    """
    Generates a single rectangle with corners in (4,2) format.
    
    If ax_aligned=True, the rectangle is axis-aligned (no rotation).
    Otherwise it has a random rotation.
    """
    # Random center
    cx = np.random.uniform(0, grid_size)
    cy = np.random.uniform(0, grid_size)

    # Random width and height
    w = np.random.uniform(5, 15)
    h = np.random.uniform(5, 15)
    
    # Half-width, half-height
    hw = w / 2.0
    hh = h / 2.0
    
    # Corners in local (unrotated) coordinates, centered at (0,0)
    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh]
    ])
    
    if ax_aligned:
        # Just shift the corners by (cx, cy)
        corners_world = corners_local + np.array([cx, cy])
    else:
        # Add a random rotation
        angle = np.random.uniform(0, 2*np.pi)
        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        corners_world = corners_local @ R.T + np.array([cx, cy])
    
    return corners_world

def generate_rectangles(n=5, grid_size=100):
    """
    Generates n rectangles on a grid of size `grid_size x grid_size`.
    By default, 3 will be axis-aligned and 2 will have random rotation.
    Returns a list of length n, each item is a (4, 2) ndarray for the rectangle corners.
    """
    rectangles = []
    
    # Let’s say 3 rectangles are axis-aligned, 2 are rotated
    num_ax_aligned = 3
    for _ in range(num_ax_aligned):
        rect = generate_random_rectangle(ax_aligned=False, grid_size=grid_size)
        rectangles.append(rect)
        
    num_rotated = n - num_ax_aligned
    for _ in range(num_rotated):
        rect = generate_random_rectangle(ax_aligned=False, grid_size=grid_size)
        rectangles.append(rect)
    
    return rectangles

def plot_points(index, points1, points2, corr, title1="Set 1", title2="Set 2", sim_dir="pybullet_env/icp/icp_plots"):
    """
    Plots two sets of points side by side on a matplotlib plot with numerical labels.

    Args:
        points1 (np.ndarray): Array of points for the first set (shape: (N, 2)).
        points2 (np.ndarray): Array of points for the second set (shape: (M, 2)).
        net_corr (list): Correspondence list for points2.
        title1 (str): Title for the first set of points.
        title2 (str): Title for the second set of points.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots side by side

    # Plot the first set of points and add labels
    ax1.scatter(points1[:, :, 0], points1[:, :, 1], label=title1)
    for i, box_points in enumerate(points1):
        avg_x = np.mean(box_points[:, 0])
        avg_y = np.mean(box_points[:, 1])
        ax1.scatter(avg_x, avg_y, color='red')  # Plot the center point
        ax1.text(float(avg_x), float(avg_y), str(i), fontsize=9, ha='center', va='bottom')  # Add numerical label
    ax1.set_title(title1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', 'box')

    # Plot the second set of points and add labels based on net_corr
    ax2.scatter(points2[:, :, 0], points2[:, :, 1], label=title2, color='orange')  # Use a different color
    for i, box_points in enumerate(points2):
        avg_x = np.mean(box_points[:, 0])
        avg_y = np.mean(box_points[:, 1])
        ax2.scatter(avg_x, avg_y, color='red')  # Plot the center point
        ax2.text(float(avg_x), float(avg_y), str(corr[i]), fontsize=9, ha='center', va='bottom')  # Add numerical label
    ax2.set_title(title2)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal', 'box')

    plt.savefig(sim_dir + f"/plot_{index}.png")  # Save the plot to a file
    plt.close()

def plot_transform(index, points1, points2, T):
    points1_flat = points1.reshape(-1, 2)
    points2_flat = points2.reshape(-1, 2)   
    m = points1_flat.shape[1]
    points1_src = np.ones((m+1, points1_flat.shape[0]))
    points1_src[:m, :] = np.copy(points1_flat.T)
    points1_src = np.dot(T, points1_src)
    points1_flat = points1_src[:points1_src.shape[0] - 1, :].T.reshape(-1, 4, 2)
    plt.figure(figsize=(6, 6))
    plt.scatter(points2_flat[:, 0], points2_flat[:, 1], color='blue', label='Points 2 (Original)')
    plt.scatter(points1_flat[:, 0], points1_flat[:, 1], color='orange', label='Points 1 (Transformed)')
    plt.title(f"Plot Transform visualized {index}")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"pybullet_env/icp/icp_plots/transform_plot_{index}.png")
    plt.close()

def main():
    if not os.path.exists("pybullet_env/icp/icp_plots"):
        os.makedirs("pybullet_env/icp/icp_plots")
    generate_points = True 
    N = 2
    num_whales = 5
    if generate_points:
        original_points = generate_rectangles()  
        points = []
        points.append(original_points)
        for _ in range(N):
            rot_theta = np.deg2rad(random.uniform(-180, 180))
            # rot_theta = 0
            # T = [0, 0]
            T = np.array([random.uniform(-50, 50), random.uniform(-50, 50)])
            R = np.array([[np.cos(rot_theta), -np.sin(rot_theta)], [np.sin(rot_theta), np.cos(rot_theta)]])
            transformed_points = []
            for box in original_points:
                transformed_box = []
                for point in box:
                    point_array = np.array(point)
                    transformed_point = np.dot(R, point_array) + T
                    transformed_box.append(transformed_point.tolist())
                random.shuffle(transformed_box)
                transformed_points.append(transformed_box)
            random.shuffle(transformed_points)
            points.append(transformed_points)
    else:
        # runs = os.listdir("pybullet_env/icp/sim_data")
        run = f"run_47" # toggle this value for custom tests
        print("Running test on:", run)
        with open(f"pybullet_env/icp/sim_data/{run}/points.json", "r") as f:
            points = json.load(f)
            N = len(points)
            num_whales = len(points[0])

    correlations = []
    for d in range(N):
        set1 = d
        set2 = (d + 1) % N
        T, corr, _ = rot_icp(np.array(points[set2]), np.array(points[set1]), N=50, use_point=False)
        correlations.append(corr)
        # plot_points(d, np.array(points[set1]), np.array(points[set2]), corr, title1=f"Set {set1}", title2=f"Set {set2}")
    correlations.reverse()
    composite = np.arange(num_whales)
    for corr in correlations:
        composite = corr[composite] 
    if composite.tolist() != [i for i in range(num_whales)]:
        print("FAILED")
        return
    print("PASSED")

if __name__ == "__main__":
    # for _ in range(20):
    main()        
