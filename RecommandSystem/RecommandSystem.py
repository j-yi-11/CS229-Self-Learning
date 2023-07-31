import numpy as np
import scipy.io as scio
train_data = scio.loadmat("./ex8_movies.mat")
print(train_data.keys())
Y = train_data['Y']
R = train_data['R']
print(Y.shape, R.shape)
avg = np.mean(Y, axis=1)


def get_loss(X, Theta, Y, R, lamb=0):
    return np.sum(((X.dot(Theta.T) - Y) * R) ** 2) / 2 + lamb * np.sum(Theta ** 2) / 2 + lamb * np.sum(X ** 2) / 2


def grad_by_x(X, Theta, Y, R, lamb=0):
    return ((X.dot(Theta.T) - Y) * R).dot(Theta) + lamb * X


def grad_by_theta(X, Theta, Y, R, lamb=0):
    return ((X.dot(Theta.T) - Y) * R).T.dot(X) + lamb * Theta

# Use the debug params to check our function
train_data = scio.loadmat("./ex8_movieParams.mat")
print(train_data.keys())
X_d = train_data['X']
Theta_d = train_data['Theta']
print(X_d.shape, Theta_d.shape)
print(get_loss(X_d[:5,:3], Theta_d[:4,:3], Y[:5,:4], R[:5,:4]))
print(get_loss(X_d[:5,:3], Theta_d[:4,:3], Y[:5,:4], R[:5,:4], lamb=1.5))
print(get_loss(X_d, Theta_d, Y, R))
print(get_loss(X_d, Theta_d, Y, R, 1))

# Gradient checking
epsilon = 1e-5
X_d_plus = np.copy(X_d)
X_d_plus[0][0] += epsilon
X_d_minus = np.copy(X_d)
X_d_minus[0][0] -= epsilon
grad_expected = (get_loss(X_d_plus, Theta_d, Y, R) - get_loss(X_d_minus, Theta_d, Y, R)) / (2 * epsilon)
print(grad_expected)
grad = grad_by_x(X_d, Theta_d, Y, R)
print(grad[0][0])
diff = np.linalg.norm(grad_expected - grad[0][0]) / np.linalg.norm(grad_expected + grad[0][0])
print(diff)


def train(Y, X, Theta, R, lr=1e-4, epochs=1000, lamb=1):
    for e in range(epochs):
        if e % 100 == 0:
            loss = get_loss(X, Theta, Y, R, lamb)
            print('loss:', loss)
        X -= lr * grad_by_x(X, Theta, Y, R, lamb)
        Theta -= lr * grad_by_theta(X, Theta, Y, R, lamb)
    return X, Theta

# Training
N = 10
num_movie, num_user = Y.shape
X = np.random.rand(num_movie, N)
Theta = np.random.rand(num_user, N)
X, Theta = train(Y, X, Theta, R)
predict = X.dot(Theta.T)
print(predict[:, 1])

import codecs
f = codecs.open("movie_ids.txt", "r", encoding = "ISO-8859-1")
lines = f.readlines()
movie_ids = {}
for line in lines:
    i = line.index(' ')
    movie_ids[int(line[:i])] = line[i:]
# Assume a new user have some rating
my_ratings = np.zeros([len(Y)])
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5
print("New user ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated {} for {}".format(my_ratings[i], movie_ids[i + 1]))

# Now add this new rating into our rating matrix
num_feat = 50
Ynew = np.c_[my_ratings, Y]
Rnew = np.c_[np.array([1 if r > 0 else 0 for r in my_ratings]), R]
print(Ynew.shape, Rnew.shape)
Xnew = np.random.standard_normal((len(Ynew), num_feat))
Tnew = np.random.standard_normal((Ynew.shape[1], num_feat))
Ymean = np.mean(Ynew, axis=1).reshape(-1, 1)
Ynorm = Ynew - Ymean
Xtrain, Ttrain = train(Ynorm, Xnew, Tnew, Rnew, lr=2e-3, epochs=500, lamb=10)
predict = Xtrain.dot(Ttrain.T)
my_predict = predict[:,0] + Ymean
print(predict[:10,0])
# Sort by rate, recommend some movies with higher score to this new user
# indexes = sorted(range(len(predict[:,0])), key=lambda i: predict[i,0], reverse=True)
indexes = np.argsort(-predict[:,0])
print("Top recommendations for you:\n")
for i in indexes[:10]:
    print("Predicting rating %.2f for movie %s" % (predict[i,0], movie_ids[i + 1]))
