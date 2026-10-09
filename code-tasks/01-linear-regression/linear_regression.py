# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

# ========== 0. 解决终端中文乱码 ==========
sys.stdout.reconfigure(encoding='utf-8')

# ========== 1. 获取脚本所在目录，保证文件生成在正确位置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'train.log')
FIG_PATH = os.path.join(SCRIPT_DIR, 'loss_curve.png')

# ========== 2. 配置日志：同时输出到终端和文件 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),   # 写文件
        logging.StreamHandler(sys.stdout)                   # 输出终端
    ]
)

# ========== 3. 生成模拟数据：y = 4 + 3x + 噪声 ==========
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

logging.info("=" * 50)
logging.info("线性回归任务开始")
logging.info(f"数据量: {len(X)} 个样本")
logging.info("真实参数: 偏置=4.00, 斜率=3.00")

# ========== 4. 从零实现梯度下降 ==========
def gradient_descent(X, y, lr=0.5, n_iters=1000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]   # 加一列1，对应偏置b
    theta = np.random.randn(2, 1)      # 随机初始化 [b, w]
    losses = []

    for i in range(n_iters):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - lr * gradients
        loss = np.mean((X_b.dot(theta) - y) ** 2)
        losses.append(loss)

        # 每100轮记录一次日志
        if (i + 1) % 100 == 0:
            logging.info(f"第{i+1}轮: loss={loss:.4f}, theta=[{theta[0][0]:.4f}, {theta[1][0]:.4f}]")

    return theta, losses

logging.info("开始训练，学习率=0.5，迭代次数=1000")
theta, losses = gradient_descent(X, y)

# ========== 5. 输出最终结果 ==========
logging.info("-" * 50)
logging.info("训练结束")
logging.info(f"学到的参数: 偏置={theta[0][0]:.4f}, 斜率={theta[1][0]:.4f}")
logging.info(f"真实参数:   偏置=4.0000, 斜率=3.0000")
logging.info(f"最终损失: {losses[-1]:.4f}")

# ========== 6. 画损失曲线 ==========
plt.figure(figsize=(8, 5))
plt.plot(losses, color='blue', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Gradient Descent Loss Curve')
plt.grid(True, alpha=0.3)
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
plt.close()

logging.info("损失曲线已保存为 loss_curve.png")
logging.info("=" * 50)