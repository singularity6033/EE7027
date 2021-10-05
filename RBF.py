from sklearn import model_selection
from scipy.io import loadmat
from math import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import numpy as np
from my_SOM.SOM import som_nn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = np.random.uniform(-1, 1, (numCenters, indim))
        self.sigma = 0.75
        self.W = np.random.random((self.numCenters, self.outdim))

    def basisfunc(self, C, D):
        assert len(D) == self.indim
        return exp(-(1 / (2 * self.sigma ** 2)) * norm(C - D) ** 2)

    def basisfunc1(self, C, D):
        assert D.shape[1] == self.indim
        return np.exp(-(1 / (2 * self.sigma ** 2)) * cdist(D, C) ** 2)

    def basisfunc2(self, C, D):
        assert len(D) == self.indim
        return exp(-(1 / (2 * self.sigma ** 2)) * norm(C - D) ** 2) * (norm(C - D) ** 2) / (self.sigma ** 3)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.centers.shape[0]), float)
        for ci, c_val in enumerate(self.centers):
            for xi, x_val in enumerate(X):
                G[xi, ci] = self.basisfunc(c_val, x_val)
        return G

    def _EstimateWeight(self, X, Y_predict, Y):
        dEdW = np.zeros((self.numCenters, Y.shape[1]), float)
        for ci, c_val in enumerate(self.centers):
            dEdW[ci, :] = np.dot(self.basisfunc1(c_val.reshape(1, -1), X).T, (Y_predict - Y))
        return dEdW

    def _EstimateCenter(self, X, Y_predict, Y):
        dEdc = np.zeros((self.numCenters, X.shape[1]), float)
        for ci, c_val in enumerate(self.centers):
            dEdc[ci, :] = np.dot(self.W[ci, :],
                                 np.dot(((Y_predict - Y) * self.basisfunc1(c_val.reshape(1, -1), X)).T,
                                        ((X - c_val) / (2 * self.sigma ** 2))))
        return dEdc

    def _EstimateSigma(self, X, Y_predict, Y):
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c_val in enumerate(self.centers):
            for xi, x_val in enumerate(X):
                G[xi, ci] = self.basisfunc2(c_val, x_val)
        dEds = np.sum(((Y_predict - Y) * np.dot(G, self.W)))
        return dEds

    def train_RS(self, X, Y):
        """
        Method #1
            This function uses randomly selection from training samples
            to construct the location of centers.
            sigma can be calculated by: d(max)/sqrt(2*m)
            X: matrix of dimensions n x indim
            Y: column vector of dimension n x 1
        """
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = X[rnd_idx, :]
        self.sigma = np.max(pairwise_distances(self.centers)) / (sqrt(2 * self.numCenters))
        # calculate activations of RBF
        G = self._calcAct(X)
        # calculate output weights (pseudo inverse)
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), Y)

    def train_PT(self, X, Y, som_h=1, som_w=1):
        """
        Method #2
            This function uses prototypes selected from training samples
            to construct the location of centers.
            som and k-means methods can be used.
            X: matrix of dimensions n x indim
            Y: column vector of dimension n x 1
            som_h: height of SOM's 2-D lattice
            som_w: height of SOM's 2-D lattice
        """
        # (i): SOM
        # self.centers = som_nn(X, som_h, som_w)
        # (ii): k-means
        k_means = KMeans(n_clusters=self.numCenters).fit(X)
        self.centers = k_means.cluster_centers_
        # calculate activations of RBF
        G = self._calcAct(X)
        # calculate output weights (pseudo inverse)
        self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), Y)

    def train_NLPB(self, X, Y, iter_num, eta1, eta2, eta3):
        """
        Method #3
            This function uses non-linear optimization method
            based on error-correction learning (using GD).
            X: matrix of dimensions n x indim
            Y: column vector of dimension n x 1
            iterations: num of iterations
            eta(1~3): learning rate or step size of weight, center location and sigma respectively.
        """
        for i in range(iter_num):
            Y_predict = self.test(X)
            # weight estimation
            dEdW = self._EstimateWeight(X, Y_predict, Y)
            # center location estimation
            dEdc = self._EstimateCenter(X, Y_predict, Y)
            # sigma estimation
            dEds = self._EstimateSigma(X, Y_predict, Y)
            # update
            self.W -= eta1 * dEdW
            self.centers -= eta2 * dEdc
            self.sigma -= eta3 * dEds

    def train_MS(self, X, Y):
        """
        Method #4
            This function uses model selection method to choose center location
            (sequential forward selection algorithm). And the stopping criterion
            is pre-defined accuracy.
            X: matrix of dimensions n x indim
            Y: column vector of dimension n x 1
        """
        numCenters = 1
        Candidates = X
        # self.W = np.empty((1, self.outdim))
        self.centers = np.zeros((1, self.indim))
        while numCenters <= self.numCenters:
            err = []
            for i in range(len(Candidates)):
                self.centers[numCenters - 1, :] = Candidates[i, :]
                # calculate activations of RBF
                G = self._calcAct(X)
                # calculate output weights (pseudo inverse)
                self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), Y)
                Y_predict = self.test(X)
                err.append(self.calcAccuracy(Y_predict, Y))
            indx = np.argmax(err)
            self.centers[numCenters - 1, :] = Candidates[indx, :]
            print(self.centers.shape)
            self.centers = np.append(self.centers, np.zeros((1, self.indim)), axis=0)
            Candidates = np.delete(Candidates, indx, axis=0)
            numCenters += 1
        self.centers = np.delete(self.centers, -1, axis=0)
        G = self._calcAct(X)
        # calculate output weights (pseudo inverse)
        self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), Y)

    def calcAccuracy(self, Y, Y_g):
        Y[Y >= 0] = 1
        Y[Y <= 0] = -1
        accuracy = len(np.where((Y - Y_g) == 0)[0]) / len(Y_g)
        return accuracy

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':
    x = loadmat("./data_train.mat")['data_train']
    y = loadmat("./label_train.mat")['label_train']
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.2)
    x_test = loadmat("./data_test.mat")['data_test']
    # rbf training process and evaluate performance
#     accuracy = np.zeros((4, 2, 32))
#     # 1st method
#     for i in range(32):
#         rbf = RBF(33, i + 2, 1)
#         rbf.train_RS(x_train, y_train)
#         y_train_predict = rbf.test(x_train)
#         y_validation_predict = rbf.test(x_validation)
#         accuracy[0, 0, i] = rbf.calcAccuracy(y_train_predict, y_train)
#         accuracy[0, 1, i] = rbf.calcAccuracy(y_validation_predict, y_validation)
#     print("done")
#     # 2nd method
#     for i in range(32):
#         rbf = RBF(33, i + 2, 1)
#         rbf.train_PT(x_train, y_train)
#         y_train_predict = rbf.test(x_train)
#         y_validation_predict = rbf.test(x_validation)
#         accuracy[1, 0, i] = rbf.calcAccuracy(y_train_predict, y_train)
#         accuracy[1, 1, i] = rbf.calcAccuracy(y_validation_predict, y_validation)
#     print("done")
#     # 3rd method
#     for i in range(32):
#         rbf = RBF(33, i + 2, 1)
#         rbf.train_NLPB(x_train, y_train, 200, 0.5, 0.5, 0.5)
#         y_train_predict = rbf.test(x_train)
#         y_validation_predict = rbf.test(x_validation)
#         accuracy[2, 0, i] = rbf.calcAccuracy(y_train_predict, y_train)
#         accuracy[2, 1, i] = rbf.calcAccuracy(y_validation_predict, y_validation)
#     print("done")
#     # 4th method
#     for i in range(32):
#         rbf = RBF(33, i + 2, 1)
#         rbf.train_MS(x_train, y_train)
#         y_train_predict = rbf.test(x_train)
#         y_validation_predict = rbf.test(x_validation)
#         accuracy[3, 0, i] = rbf.calcAccuracy(y_train_predict, y_train)
#         accuracy[3, 1, i] = rbf.calcAccuracy(y_validation_predict, y_validation)
# # print("accuracy is {:.2%}".format())
# #
# x = np.linspace(2, 33, 32)
# print(x)
# plt.figure(1)
# plt.plot(x, accuracy[0, 0, :], marker='o')
# plt.plot(x, accuracy[1, 0, :], marker='v')
# plt.plot(x, accuracy[2, 0, :], marker='x')
# plt.plot(x, accuracy[3, 0, :], marker='s')
# plt.grid()
# plt.ylim([0, 1])
# plt.title("Accuracy of training set")
# plt.xlabel("Number of neurons in the hidden layer")
# plt.ylabel("Accuracy")
# plt.legend(['Method 1', 'Method 2', 'Method 3', 'Method 4'], loc="best")
# plt.savefig('rbf1.jpg')
# plt.figure(2)
# plt.plot(x, accuracy[0, 1, :], marker='o')
# plt.plot(x, accuracy[1, 1, :], marker='v')
# plt.plot(x, accuracy[2, 1, :], marker='x')
# plt.plot(x, accuracy[3, 1, :], marker='s')
# plt.grid()
# plt.ylim([0, 1])
# plt.title("Accuracy of validation set")
# plt.xlabel("Number of neurons in the hidden layer")
# plt.ylabel("Accuracy")
# plt.legend(['Method 1', 'Method 2', 'Method 3', 'Method 4'], loc="best")
# plt.savefig('rbf2.jpg')
rbf = RBF(33, 33, 1)
rbf.train_MS(x_train, y_train)
y_test = rbf.test(x_test)
