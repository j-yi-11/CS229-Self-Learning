import numpy as np
train_data = np.loadtxt("./ex1data2.txt", delimiter=",")
X = train_data[:, :2]
Y = train_data[:, 2:]
# Normalize
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
X = np.c_[np.ones(len(X)), X]
# Predict using one testing item
test_x = np.array([1650, 3]).reshape(1, 2)
test_x = (test_x - mean) / std
test_x = np.c_[np.ones(len(test_x)), test_x]
# Normal Equation is very good, but the performance may be worst when the training set is large
theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
print('Theta:', theta)
predict1 = np.matmul(test_x, theta)
print(predict1)
