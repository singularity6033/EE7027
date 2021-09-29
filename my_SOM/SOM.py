import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import math


def distance_matrix(n1, n2):
    # this function calculate the lateral distance between neurons in the 2D lattice
    # Dist(i, j) denotes the distance between neurons i and j
    # pairwise_distances()
    # define the position of all neurons in the 2D lattice
    posit = np.zeros((n1 * n2, 2))
    for i in range(1, n1 + 1):
        posit[(i - 1) * n2:i * n2 - 1, 0] = i - 1
        posit[(i - 1) * n2:i * n2 - 1, 1] = np.arange(n2 - 1)
    return cdist(posit, posit)


def som_nn(data, n1, n2):
    # input
    # data: a matrix and each row is one sample
    # n1, n2: the no. of rows and columns of the 2D lattice
    # output
    # w: weight of the neurons

    # No. of samples, dimensionality of input space anf total No. of neurons
    nSample = np.size(data, 0)
    nDim = np.size(data, 1)
    nNeuron = n1 * n2

    # generate the initial weight vectors
    w = np.random.randn(nNeuron, nDim)

    # define initial values for the time constant and the learning rate
    eta0 = 0.1
    sigma0 = 0.5 * math.sqrt(((n1 - 1) ** 2 + (n2 - 1) ** 2))
    tau1 = 1000 / math.log(sigma0)

    # generate the latest distance matrix
    Dist = distance_matrix(n1, n2)

    # The self-organizing phase
    for k in range(1, 1001):
        # calculate the learning rate and width of the neighborhood function at current iteration
        eta = eta0 * math.exp(-k / 1000)
        sigma = sigma0 * math.exp(-k / tau1)

        # randomly select a training sample
        temp = np.random.permutation(nSample) - 1
        j = temp[0]
        x = data[j, :]
        d = np.zeros(nNeuron)

        # compute and find the winning neuron
        for i in range(0, nNeuron):
            d[i] = np.dot((w[i, :] - x), (w[i, :] - x).T)
        xx = np.min(d)
        index_win = np.argmin(d)

        # update weight vectors of all neurons
        for i in range(0, nNeuron):
            h = math.exp(-Dist[i, index_win] / 2 / (sigma ** 2))
            w[i, :] += eta * h * (x - w[i, :])

    # the convergence phase
    # set the learning rate to a small constant
    eta = 0.01

    # repeat 500*nNeuron times
    for k in range(1, 500 * nNeuron + 1):
        # randomly select a training sample
        temp = np.random.permutation(nSample) - 1
        j = temp[5]
        x = data[j, :]

        # compute and find the winning neuron
        for i in range(0, nNeuron):
            d[i] = np.dot((w[i, :] - x), (w[i, :] - x).T)
        xx = np.min(d)
        index_win = np.argmin(d)

        # update the weight vector of the winning neuron only
        h = 1
        w[index_win, :] += eta * h * (x - w[index_win, :])
    return w


# input_x = np.random.rand(1000, 2)
# plt.subplot(121)
# plt.scatter(input_x[:, 0], input_x[:, 1])
# w = som_nn(input_x, 5, 6)
# plt.subplot(122)
# plt.scatter(w[:, 0], w[:, 1])
# plt.show()
