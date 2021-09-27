from sklearn import *
from scipy.io import loadmat
from math import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import numpy as np
from my_SOM.SOM import som_nn
from sklearn.cluster import KMeans


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = np.random.uniform(-1, 1, (numCenters, indim))
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))

    def basisfunc(self, C, D):
        assert len(D) == self.indim
        return exp(-self.beta * norm(C - D) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c_val in enumerate(self.centers):
            for xi, x_val in enumerate(X):
                G[xi, ci] = self.basisfunc(c_val, x_val)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
        y: column vector of dimension n x 1 """
        # # method *1: choose random center vectors from training set
        # rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        # self.centers = X[rnd_idx, :]
        # method *2: choose prototypes of training samples
        # # (i): SOM
        # self.centers = som_nn(X, 5, 4)
        # (ii): K-means
        k_means = KMeans(n_clusters=self.numCenters).fit(X)
        self.centers = k_means.cluster_centers_
        # print(type(self.centers))
        # calculate activations of RBFs
        G = self._calcAct(X)
        # calculate output weights (pseudo inverse)
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':
    x = loadmat("./data_train.mat")['data_train']
    y = loadmat("./label_train.mat")['label_train']
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3)
    x_test = loadmat("./data_test.mat")['data_test']
    # rbf
    rbf = RBF(33, 20, 33)
    rbf.train(x_train, y_train)
    z = rbf.test(x_validation)
    z[z > 0] = 1
    z[z < 0] = -1
    temp = np.where((z - y_validation) == 0)
    accuracy = len(temp[0]) / len(y_validation)
    print("validation accuracy is {:.2%}".format(accuracy))
