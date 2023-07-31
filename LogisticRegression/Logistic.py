import numpy as np
import matplotlib.pyplot as plt
train_data = np.loadtxt("./ex2data1.txt", delimiter=",", encoding='utf-8')
# 取出正负数据
positives = train_data[train_data[:, -1] == 1]
# train[:, -1]， 是说对train这个二维的数据，逗号分隔开的前面的":"是说取全部的行，逗号后面的-1是说取最后一列。
negatives = train_data[train_data[:, -1] == 0]
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Distribution')
plt.xlabel('exam1 score')
plt.ylabel('exam2 score')
# 默认 1 需要 ； 0 不需要
plt.scatter(positives[:, 0], positives[:, 1], color='black', marker='+')
plt.scatter(negatives[:, 0], negatives[:, 1], color='red', marker='o')
plt.legend(('Admitted', 'Not admitted'), loc='best')
plt.show()


# Regularized logistic regression
train_data = np.loadtxt("./ex2data2.txt", delimiter=",")
print(train_data[:5,:])
positives = train_data[train_data[:, -1] == 1]
negatives = train_data[train_data[:, -1] == 0]
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Distribution')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.scatter(positives[:, 0], positives[:, 1], color='black', marker='+')
plt.scatter(negatives[:, 0], negatives[:, 1], color='red', marker='o')
plt.legend(('y=1', 'y=0'), loc='best')

plt.show()
Y = np.c_[train_data[:, -1]]


# Polynominal features
def map_features(data):
    data = np.c_[np.ones(len(data)), data]
    for i in range(2, 6 + 1):
        for j in range(0, i + 1):
            new_feature = ((data[:, 1]**j) * (data[:, 2])**(i - j)).reshape(-1, 1)
    #         print(new_feature.shape)
            data = np.hstack((data, new_feature))
    return data

X = map_features(train_data[:, :-1])

print(X[:1, :])


def hypothesis(X, theta):
    return 1 / (1 + np.exp(-np.matmul(X, theta)))


def compute_loss(X, Y, theta):
    H = hypothesis(X, theta)
    return np.sum(-np.matmul(Y.T, np.log(H)) - np.matmul((1 - Y).T, np.log(1 - H))) / len(X)


# Training
X = train_data[:, :-1]
X = np.c_[np.ones(len(X)), X]
Y = np.c_[train_data[:, -1]]
theta = np.zeros([3, 1])
epoch = 10
# learning_rate = 0.001
losses = np.array([])
for i in range(epoch):
    losses = np.append(losses, compute_loss(X, Y, theta))
    H = hypothesis(X, theta)
    ## Gradient descend is very slow
    #     delta = np.matmul((H - Y).T, X) / len(X)
    #     theta -= learning_rate * delta.T
    # Use Newton method instead
    gradient = np.matmul((H - Y).T, X) / len(X)
    #     print(gradient)
    # Refer https://zhuanlan.zhihu.com/p/63305895 to know how to compute Hessian matrix
    hessian = np.matmul(np.matmul(X.T, np.diag(H.ravel() * (1 - H).ravel())), X) / len(X)
    #     print(hessian)
    delta = np.matmul(gradient, np.linalg.inv(hessian))
    theta -= delta.T

print(losses[0], losses[-1])
# 测试一个例子
test_x = np.array([1, 45, 85])
predict1 = hypothesis(test_x, theta)
print(predict1)

# Plot the decision boundary
positives = train_data[train_data[:, -1] == 1]
negatives = train_data[train_data[:, -1] == 0]
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Distribution')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.scatter(positives[:, 0], positives[:, 1], color='black', marker='+')
plt.scatter(negatives[:, 0], negatives[:, 1], color='red', marker='o')
plt.legend(('Admitted', 'Not admitted'), loc='best')
x1 = np.arange(20, 100, 0.5)
x2 = (-theta[0] - theta[1] * x1) / theta[2]
plt.plot(x1, x2)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(len(losses)), losses)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Distribution')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.scatter(positives[:, 0], positives[:, 1], color='black', marker='+')
plt.scatter(negatives[:, 0], negatives[:, 1], color='red', marker='o')
plt.legend(('y=1', 'y=0'), loc='best')
x1 = np.linspace(X[:, 1].min(), X[:, 1].max())
x2 = np.linspace(X[:, 2].min(), X[:, 2].max())
xx1, xx2 = np.meshgrid(x1, x2)
print(map_features(np.c_[x1, x2]).ravel().shape, theta.shape)
h = hypothesis(map_features(np.c_[xx1.ravel(), xx2.ravel()]), theta)
plt.contour(xx1, xx2, h.reshape(xx1.shape), [0.5], colors='black', linewidths=.5)
plt.show()
