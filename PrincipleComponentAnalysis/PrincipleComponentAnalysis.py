import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
# PCA
train_data = scio.loadmat("./ex7data1.mat")
X = train_data['X']
print(X.shape)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Data distribution')
plt.xlabel('')
plt.ylabel('')
plt.scatter(X[:,0], X[:,1], color='green', marker='o')
plt.show()


def normalize(X):
    return X - np.mean(X, axis=0)


def pca(X, k):
    Sigma = X.T.dot(X) / len(X)# Covarriance matrix
    U, S, V = np.linalg.svd(Sigma)
    Z = X.dot(U[:,:k])
    X_approx = Z.dot(U[:,:k].T)
    return X_approx

# Normalize first in order to compute the covariance matrix
X = normalize(X)
X_approx = pca(X, 1)
print(X_approx[0])


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Data distribution')
plt.xlabel('')
plt.ylabel('')
# Original points
plt.scatter(X[:, 0], X[:, 1], color='green', marker='o')
# Plot the low dimention points
plt.scatter(X_approx[:,0], X_approx[:, 1], color='red', marker='x')
# Plot the projection direction
plt.plot([X[:,0], X_approx[:,0]], [X[:,1], X_approx[:,1]], linestyle='--', color='blue')
plt.show()

from PIL import Image
train_data = scio.loadmat("./ex7faces.mat")
X = train_data['X']
print(X.shape)
fig = plt.figure(figsize=(10, 10))
X = X[:100, :]
SIZE = 10
fig, ax_array = plt.subplots(SIZE, SIZE, sharey=True, sharex=True, figsize=(SIZE, SIZE))
for i in range(0, SIZE):
    for j in range(0, SIZE):
        arr = X[i * SIZE + j].reshape((32, 32), order='F')# order 'F' is Fortan-Style!
        ax_array[i, j].matshow(arr, cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()
X = normalize(X)
X_approx = pca(X, 100)
print(X_approx.shape)

# Plot the data using low dimentation images
fig = plt.figure(figsize=(10, 10))
SIZE = 10
fig, ax_array = plt.subplots(SIZE, SIZE, sharey=True, sharex=True, figsize=(SIZE, SIZE))
for i in range(0, SIZE):
    for j in range(0, SIZE):
        arr = X_approx[i * SIZE + j].reshape((32, 32), order='F')
        ax_array[i, j].matshow(arr, cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()
