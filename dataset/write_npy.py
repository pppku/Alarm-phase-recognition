import numpy as np

# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4]])

# 保存.npy文件
np.save("../data/arr.npy", arr)