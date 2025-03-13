import numpy as np

correlations = [[4, 3, 1, 2, 0], [3, 4, 0, 1, 2], [1, 4, 0, 3, 2]]    

net_corrs = [] # net_corrs[i] is the mapping from point cloud i to point cloud 0

for i in range(len(correlations)):
    composite = np.arange(len(correlations[i]))
    for j in range(i, -1, -1):
        new_composite = np.zeros(len(composite), dtype=int)
        for k in range(len(composite)):
            new_composite[correlations[j][k]] = composite[k]
        composite = new_composite.copy()
    net_corrs.append(composite)

print(net_corrs)


