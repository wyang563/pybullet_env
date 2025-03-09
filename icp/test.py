import numpy as np

box1 = np.array([[1, 1], [4, 4], [0, 1], [2, 3]])
box2 = np.array([[2, 2], [5, 5], [1, 2], [3, 4]])

center1 = np.mean(box1, axis=0)
center2 = np.mean(box2, axis=0)
print(center1, center2)