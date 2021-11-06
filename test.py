import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.linalg import norm, pinv
from scipy.io import loadmat

a = [1, 2, 3]
ac = []
ac = a[:2]
ac = ac[::-1]
a[:2] = ac
# print([x for x in a if x==2])
print(a)
# a = np.random.uniform(0, 5, (2, 2))
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]])
# print(np.sum(a,axis=-1))
# mat1 = np.expand_dims(a, 0).repeat(a.shape[0], 0)
# x = loadmat("./data_train.mat")['data_train']
# for i in range(30):
#     rnd_idx = np.random.permutation(x.shape[0])[:i+1]
#     print(np.max(pairwise_distances(x[rnd_idx, :])))
# c = np.zeros((1, 3))
# c[0, :] = np.array([1, 2, 3])
# a = np.array([[1, 2], [3, 4], [5, 6]])
# b = np.array([[1, 3, 4]])
# e = [1,2,3,4]
# d = np.empty((1,2))
# d = np.append(d, b, axis=0)
# b[, :] = np.array([[8, 3]])
# print(b.shape)
# print(np.append(c,b,axis=0))
# b = np.expand_dims(a, -1)
# print(np.array([[1,2,3]]).shape)

# print(mat1.shape)

# == pairwise_distance
# def matrix_dist(inputs):
#     """
#     calculate distance(euclidean) of every elements of a matrix
#     param mat: matrix with n*m
#     return: size of n*n distance matrix
#     """
#     temp1 = np.expand_dims(inputs, 1)
#     temp2 = np.expand_dims(inputs, 0)
#     return np.sqrt(np.sum((temp1 - temp2) ** 2, axis=-1))
#
#
# print(matrix_dist(a))
