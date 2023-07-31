import numpy as np
import matplotlib.pyplot as plt
#np.eye(5)创建5*5单位矩阵
# 读入训练集train_data，训练集名字为ex1data1.txt,保存在本文件同一个文件夹内，训练集的数据用delimiter','隔开
train_data = np.loadtxt("./ex1data1.txt", delimiter=",")
#给训练集train_data可视化
populations = train_data[:, 0]#population是train_data第1列组成的列向量
profits = train_data[:, 1]#profit是train_data第2列组成的列向量
m = len(profits)# m 数据个数  len(a)  返回矩阵a的列数（向量个数）
print('the number of the elements of the training set is :')
print(m)
#开始画图
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#add_subplot(1, 1, 1) 画布分成1*1，ax1占第1块
# 标题，横轴纵轴文字信息
ax1.set_title('the original data of profit and population distribution')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.scatter(populations, profits, color='red', marker='x')
plt.show()


def hypothesis(X, theta):
    return np.matmul(X, theta)
# np.matmul(a,b)  矩阵乘法a*b
def compute_loss(X, Y, theta):
    H = hypothesis(X, theta)
    return np.sum((H - Y) ** 2 / (2 * len(X)))
#损失函数定义
# Add all 1s to the first column as bias
# np.c_[a,b] -- 把矩阵a,b按照行数一样的规则拼接为 [a ; b]
X = np.c_[np.ones(len(train_data)), train_data[:, 0]]
Y = np.c_[train_data[:, 1]]
theta = np.zeros([2, 1])#  h(x,θ) = θ0 + θ1*x 一共两项
epoch = 1500# 训练次数 1500 次
learning_rate = 0.001
losses = np.array([])
for i in range(epoch): # range(n) 返回1-n等差数列
    losses = np.append(losses, compute_loss(X, Y, theta))
# np.append(a,b) a,b 展平成一维数组后 b 拼接到 a 后面
    H = hypothesis(X, theta)
    delta = learning_rate * (np.matmul((H - Y).T, X) / len(X))
    theta -= delta.T    # .T    矩阵转置

# 自己用两组样例试一试
predict1 = np.matmul([1, 3.5], theta);
print(predict1)
predict2 = np.matmul([1, 7], theta);
print(predict2)

# Plot the trained model
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('profit and population distribution')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.scatter(populations, profits, color='red', marker='x')
plt.plot(train_data[:, 0], hypothesis(X, theta))
plt.legend(('Line regression', 'Training Data'), loc='best')# 图例内容 位置loc
plt.show()

# Plot the loss
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(len(losses)), losses)
plt.show()

# Data to plot 3D surface and contour graph
from mpl_toolkits.mplot3d import Axes3D
size = 100
# np.linspace(start,end,num)  在start和end中间构造num个数组成的等差数列
theta0 = np.linspace(-10, 10, size)
theta1 = np.linspace(-1, 4, size)
losses = np.zeros((len(theta0), len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = np.array([theta0[i], theta1[j]]).reshape((2, -1))
        loss = compute_loss(X, Y, t)
        losses[i, j] = np.sum(loss)
# We need mesh grid
theta0, theta1 = np.meshgrid(theta0, theta1)# np.meshgrid 生成网格点矩阵



# Plot the surface graph for loss
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Loss - Surface graph')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta)$')
ax.plot_surface(theta0, theta1, losses, cmap = plt.get_cmap('rainbow'))
plt.show()


# Plot the contour for loss
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title('Contour graph')
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'$\theta_1$')
contour = ax2.contour(theta0, theta1, losses, np.logspace(-1,2,15))
plt.clabel(contour, inline=1, fontsize=10)
# print(theta)
plt.scatter([theta[0,0]], [theta[1,0]], color='red', marker='x')
plt.show()


