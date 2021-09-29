from sklearn import svm, model_selection
from scipy.io import loadmat

x = loadmat("./data_train.mat")['data_train']
y = loadmat("./label_train.mat")['label_train']
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.4)
x_test = loadmat("./data_test.mat")['data_test']

clf = svm.SVC(kernel='rbf', gamma='auto', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
score = clf.score(x_validation, y_validation.ravel())
print("accuracy is {:.2%}".format(score))
y_test = clf.predict(x_test)