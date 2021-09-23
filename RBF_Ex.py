import numpy as np

theta = 0.707
input_x = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
c1 = np.array([1, 1])
c2 = np.array([0, 0])
print(input_x - c1)
temp1 = np.exp(-np.sum(((input_x - c1) ** 2), axis=1) / (2 * theta ** 2))
temp2 = np.exp(-np.sum(((input_x - c2) ** 2), axis=1) / (2 * theta ** 2))
w0 = np.array([1, 1, 1, 1])
b = np.column_stack((temp1, temp2, w0.T))
d = np.array([0, 1, 0, 1])

w = np.dot(np.dot(np.linalg.inv(np.dot(b.T, b)), b.T), d.T)
f = np.dot(b, w)
print(f)

