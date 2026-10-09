import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据：y = 4 + 3x + 噪声
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. 从零实现梯度下降
def gradient_descent(X, y, lr=0.1, n_iters=1000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]   # 加一列1，对应偏置b
    theta = np.random.randn(2, 1)      # 随机初始化参数 [b, w]
    losses = []
    
    for i in range(n_iters):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr * gradients
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        losses.append(loss)
    
    return theta, losses

theta, losses = gradient_descent(X, y)
print(f"学到的参数: 偏置={theta[0][0]:.2f}, 斜率={theta[1][0]:.2f}")
print(f"真实参数: 偏置=4, 斜率=3")

# 3. 画损失曲线
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Gradient Descent Loss Curve')
plt.savefig('loss_curve.png')
plt.show()