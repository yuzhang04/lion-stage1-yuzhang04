# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# ========== 0. 环境配置与日志设置 ==========
# 解决终端中文乱码
sys.stdout.reconfigure(encoding='utf-8')

# 自动获取当前脚本所在目录，保证文件生成在正确位置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'train.log')

# 配置日志：同时输出到终端和 train.log 文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'), # 写进文件（mode='w'：每次运行覆盖，避免历史记录堆积）
        logging.StreamHandler(sys.stdout)                # 打印到控制台
    ]
)
# ========== 1. 生成合成数据 ==========
np.random.seed(100) # 固定随机种子，保证每次运行结果一样
X = 2 * np.random.rand(100, 1) # 生成 100 个 x，范围在 0-2 之间，rand 代表生成 0 到 1 之间的随机小数，(100, 1) 代表生成一个矩阵形状：100 行，1 列
# 真实关系: y = 4 + 3x + 高斯噪声，模拟真实情况
y = 4 + 3 * X + np.random.randn(100, 1)

# ========== 2. 准备数据矩阵 ==========
m = X.shape[0] # 样本数（从数据本身取，不要硬编码，改数据量时不会漏改）
# 在 X 左边拼接一列全 1，用于计算截距 (bias)，np.c_[]：这是 NumPy 的拼接语法（Column 拼接）
X_b = np.c_[np.ones((m, 1)), X] 

# ========== 3. 初始化参数 ==========
theta = np.zeros((2, 1)) # 零初始化参数 [b, w] (即 [截距, 斜率])，与后续任务统一，便于复现
learning_rate = 0.1
n_iterations = 100

losses = []
# 记录训练开始信息
logging.info("=" * 50)
logging.info("开始训练线性回归模型...")
logging.info(f"数据量: {m} 个样本, 学习率: {learning_rate}, 迭代次数: {n_iterations}")
logging.info("真实参数: 截距=4.00, 斜率=3.00")
# ========== 4. 梯度下降循环 ==========
for iteration in range(n_iterations):
    #计算预测值，即假设函数（前向传播）
    predictions=X_b.dot(theta)
    #梯度下降
    gradient=2/m*X_b.T.dot(predictions-y)
    theta=theta-learning_rate*gradient
    #重新计算 predictions 保证不滞后
    predictions_new=X_b.dot(theta)
    #计算损失
    loss=np.mean((predictions_new-y)**2)
    losses.append(loss)

    # 每 100 轮往日志里打一次卡
    if (iteration + 1) % 100 == 0:
        logging.info(f"第 {iteration+1:4d} 轮迭代: Loss = {loss:.4f}")

logging.info(f"梯度下降最终参数: 截距={theta[0][0]:.3f}, 斜率={theta[1][0]:.3f}")

# ========== 4. 与正规方程交叉验证（本任务最关键的一步）==========
# 用正规方程对比验证
theta_pinv = np.linalg.pinv(X_b).dot(y)
logging.info(f"正规方程精确解: 截距={theta_pinv[0][0]:.3f}, 斜率={theta_pinv[1][0]:.3f}")
# 断言 1：如果两者差异过大，程序会报错
assert np.allclose(theta, theta_pinv, atol=0.1), "梯度下降没收敛！"
logging.info(f"与精确解的最大差: {np.abs(theta - theta_pinv).max():.5f}（判据 < 0.1）")

# 断言 2：损失必须全程单调下降（容差 1e-12 用于避开浮点噪声；
#         1000 轮时会浮出 6.7e-16 级别的"回升"，那是浮点误差不是 bug）
max_rise = float(np.diff(losses).max())
assert max_rise <= 1e-12, f"损失曲线出现回升: {max_rise:.2e}"
logging.info(f"损失全程单调下降：OK（最大回升 {max(max_rise, 0.0):.2e}）")

# ========== 5. 可视化 ==========
plt.figure(figsize=(12, 5))

# 图 1：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('True MSE Loss Curve') 

# 图 2：拟合直线
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data Points')

xs = np.linspace(0, 2, 200)
ys = theta[0, 0] + theta[1, 0] * xs
plt.plot(xs, ys, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'loss_curve.png'), dpi=150)
plt.show()