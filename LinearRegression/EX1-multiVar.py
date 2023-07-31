import numpy as np
import matplotlib.pyplot as plt
train_data = np.loadtxt("./ex1data2.txt", delimiter=",")
X = train_data[:, :2]
Y = train_data[:, 2:]
# Normalize
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
X = np.c_[np.ones(len(X)), X]
# Training


def hypothesis(X, theta):
    return np.matmul(X, theta)


def compute_loss(X, Y, theta):
    H = hypothesis(X, theta)
    return np.sum((H - Y) ** 2 / (2 * len(X)))


theta = np.zeros([X.shape[1], 1])
epoch = 1500
learning_rate = 0.01
losses = np.array([])
for i in range(epoch):
    losses = np.append(losses, compute_loss(X, Y, theta))
    H = hypothesis(X, theta)
    delta = learning_rate * (np.matmul((H - Y).T, X) / len(X))
    theta -= delta.T
print('Theta:', theta)

# Predict using one testing item
test_x = np.array([1650, 3]).reshape(1, 2)
test_x = (test_x - mean) / std
test_x = np.c_[np.ones(len(test_x)), test_x]
predict1 = np.matmul(test_x, theta)
print(predict1)
# Plot the loss
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.plot(range(len(losses)), losses)

plt.show()
