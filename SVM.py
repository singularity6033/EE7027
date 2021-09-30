from sklearn import svm, model_selection
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

x = loadmat("./data_train.mat")['data_train']
y = loadmat("./label_train.mat")['label_train']
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.4)
x_test = loadmat("./data_test.mat")['data_test']

gamma = 0.1
train_score = []
validation_score = []
svm_obj = []
# gamma ranges from 0.1~1 and evaluate the performance
for i in range(10):
    clf = svm.SVC(kernel='rbf', gamma=gamma, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    svm_obj.append(clf)
    train_score.append(clf.score(x_train, y_train))
    validation_score.append(clf.score(x_validation, y_validation))
    gamma += 0.1
    gamma = round(gamma, 1)

clf = svm.SVC(kernel='rbf', gamma="auto", decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
train_score_auto = clf.score(x_train, y_train)
validation_auto = clf.score(x_validation, y_validation)

x = np.linspace(0.1, 1, 10)
plt.plot(x, train_score, x, validation_score)
plt.scatter(1/x_train.shape[0], train_score_auto,  marker="x")
plt.scatter(1/x_train.shape[0], validation_auto, marker="x")
plt.title("Accuracy of training and validation set")
plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1])
plt.legend(['training set', 'validation set', 'training set with gamma=1/n_features',
            'validation set with gamma=1/n_features'], loc="best")
plt.show()
# plt.savefig('svm.jpg')

y_test = svm_obj[1].predict(x_test)
