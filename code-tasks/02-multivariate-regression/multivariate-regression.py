# -*- coding: utf-8 -*-
import sys
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

# 中文显示：Windows 用微软雅黑 / 黑体，否则图上的中文会变成方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 0. 环境与日志配置 ==========
sys.stdout.reconfigure(encoding='utf-8')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'train.log')

#每次运行重写日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ========== 1. 生成数据（特征量级故意拉开 20 倍）==========
np.random.seed(42)
m = 100
x1 = np.random.rand(m, 1) * 100      # 范围 0-100，std ≈ 29.6
x2 = np.random.rand(m, 1) * 5        # 范围 0-5，  std ≈ 1.46  -> 量级差 20 倍
y = 5 + 2 * x1 + 3 * x2 + np.random.randn(m, 1) * 5
X = np.c_[x1, x2]
theta_true = np.array([5.0, 2.0, 3.0])     # [b, w1, w2]

# ========== 2. 缩放 X 和 y ==========
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)   # 逐列算统计量
X_scaled = (X - X_mean) / X_std                         # Z-score 标准化
y_mean, y_std = np.mean(y), np.std(y)
y_scaled = (y - y_mean) / y_std

X_b = np.c_[np.ones((m, 1)), X]                 # 原始尺度设计矩阵
X_scaled_b = np.c_[np.ones((m, 1)), X_scaled]   # 缩放尺度设计矩阵


# ========== 3. 工具函数 ==========
def to_original(theta_s):
    "把「缩放空间」的参数还原回原始尺度。"
    c = theta_s[0, 0]#取出缩放空间的截距 c
    a = theta_s[1:, 0]#取出缩放空间的所有特征斜率
    w = y_std * a / X_std                       # 斜率还原
    b = y_mean + y_std * c - np.sum(w * X_mean) # 截距还原
    return np.array([b, w[0], w[1]])


def mse_orig(theta):
    " cost function:MSE = mean((X_b·theta - y)^2)"
    err = X_b.dot(theta) - y
    return float(np.mean(err ** 2))


def gradient_descent(Xm, yv, learning_rate, n_iterations, scaled=False):
    "梯度下降。scaled=True 表示在缩放空间迭代，但每轮都把参数还原回原始尺度再记 loss "
    theta = np.zeros((Xm.shape[1], 1))
    losses = np.empty(n_iterations)
    for i in range(n_iterations):
        predictions = Xm.dot(theta)
        gradients = 2 / m * Xm.T.dot(predictions - yv)
        theta = theta - learning_rate * gradients          # 同步更新
        th_eval = to_original(theta).reshape(-1, 1) if scaled else theta
        losses[i] = mse_orig(th_eval)
    th_final = to_original(theta).reshape(-1, 1) if scaled else theta
    return th_final, losses


def normal_equation(Xm, yv):
    "正规方程解：theta = (X^T X)^(-1) X^T y"
    theta = np.linalg.pinv(Xm.T.dot(Xm)).dot(Xm.T).dot(yv)
    return theta


def first_reach(losses, tol):
    "第一次达到 tol 的轮数（1-based）；始终没达到返回 None"
    idx = int(np.argmax(losses <= tol))
    return idx + 1 if losses[idx] <= tol else None


# ========== 4. 运行对比 ==========
logging.info("=" * 62)
logging.info("开始训练：多元线性回归三法对比")

# 先算理论参照：正规方程解处的 MSE，即这 100 个样本能达到的最优值
theta_ne = normal_equation(X_b, y)
J_star = mse_orig(theta_ne)
tol = J_star * 1.01
logging.info(f"样本最优 MSE J* = {J_star:.4f}（正规方程解处）")
logging.info(f"收敛判据：MSE <= {tol:.4f}（J* 的 1.01 倍）")

# ---- 为什么无缩放这么慢：条件数 ----
cond_raw = np.linalg.cond(X_b.T.dot(X_b))
cond_scl = np.linalg.cond(X_scaled_b.T.dot(X_scaled_b))
lam_max_raw = np.linalg.eigvalsh(2 * X_b.T.dot(X_b) / m).max()      # MSE 口径 Hessian
lam_max_scl = np.linalg.eigvalsh(2 * X_scaled_b.T.dot(X_scaled_b) / m).max()
logging.info("-" * 62)
logging.info(f"条件数 κ(X^T X)：未缩放 {cond_raw:.1f}  ->  缩放后 {cond_scl:.2f}")
logging.info(f"α 稳定上限 1/λmax：未缩放 {1/lam_max_raw:.6f}  ->  缩放后 {1/lam_max_scl:.4f}")
logging.info("（λmax ∝ 特征尺度²，x1 放大 20 倍 -> α 上限缩小约 400 倍，这就是'相乘关系'）")

# ---- A. 无缩放 GD：α 被迫压到 1e-4，仍要 3.7 万轮 ----
LR_RAW, IT_RAW = 1e-4, 40000
theta_raw, losses_raw = gradient_descent(X_b, y, LR_RAW, IT_RAW, scaled=False)
n_raw = first_reach(losses_raw, tol)

# ---- B. 有缩放 GD：α=0.1，20 多轮到位 ----
LR_SCL, IT_SCL = 0.1, 300
theta_scl, losses_scl = gradient_descent(X_scaled_b, y_scaled, LR_SCL, IT_SCL, scaled=True)
n_scl = first_reach(losses_scl, tol)

# ---- C. 正规方程：一步到位 ----
loss_ne = mse_orig(theta_ne)

results = [
    ("无缩放 GD", LR_RAW, IT_RAW, n_raw, theta_raw, losses_raw),
    ("有缩放 GD", LR_SCL, IT_SCL, n_scl, theta_scl, losses_scl),
    ("正规方程",  None,   1,      1,     theta_ne,   np.array([loss_ne])),
]

logging.info("-" * 62)
for name, lr, budget, n, th, L in results:
    it = "未达标" if n is None else f"{n}"
    lr_str = "-" if lr is None else f"{lr:g}"
    logging.info(f"{name}  α={lr_str:<8} 达标轮数={it:<8} 最终MSE={L[-1]:.4f} "
                 f"参数=[{th[0,0]:.3f}, {th[1,0]:.3f}, {th[2,0]:.3f}]")
logging.info(f"真实参数          [5.000, 2.000, 3.000]（注意：正规方程也只回到样本最优，不等于真值）")

# ---- 附加验证：正规方程「不需要」缩放 ----
theta_ne_from_scaled = to_original(normal_equation(X_scaled_b, y_scaled)).reshape(-1, 1)
gap = float(np.abs(theta_ne - theta_ne_from_scaled).max())
logging.info(f"[验证] 正规方程：原始 X 求解 vs 缩放 X 求解再还原，最大差 = {gap:.2e}（应≈0，说明不迭代就不受尺度影响）")

# ---- 参数离真值多远？用标准误检验是否合理 ----
sigma = 5.0
se = np.array([np.nan, sigma / (X_std[0] * np.sqrt(m)), sigma / (X_std[1] * np.sqrt(m))])
logging.info(f"[抽样误差] w1 标准误 ≈ {se[1]:.3f}，w2 标准误 ≈ {se[2]:.3f}"
             f"（w2 与真值差 {abs(theta_ne[2,0]-3.0):.3f} ≈ {abs(theta_ne[2,0]-3.0)/se[2]:.1f} 个标准误，属正常抽样波动）")


# ========== 5. 对比结果表（任务书要求产出）==========
lines = []
lines.append("# 代码任务②：三法对比结果表\n")
lines.append(f"- 数据：`y = 5 + 2·x1 + 3·x2 + N(0,5²)`，m=100；x1∈[0,100]（std≈29.6），x2∈[0,5]（std≈1.46）")
lines.append(f"- 统一口径：loss 与参数**全部在原始尺度**评估；收敛判据 MSE ≤ J*×1.01，J* = {J_star:.4f}\n")
lines.append("| 方法 | α | 迭代预算 | 达标轮数 | 相对有缩放 | 最终 MSE | 参数 [b, w1, w2] | 与正规方程解最大差 |")
lines.append("|---|---|---|---|---|---|---|---|")
base = n_scl
for name, lr, budget, n, th, L in results:
    ratio = "-" if n is None else f"{n / base:.0f}×"
    lr_str = "-" if lr is None else f"{lr:g}"
    it = f"{n:,}" if n is not None else f">{budget:,}"
    gap_ne = float(np.abs(th - theta_ne).max())
    lines.append(f"| {name} | {lr_str} | {budget:,} | {it} | {ratio} | {L[-1]:.4f} | "
                 f"[{th[0,0]:.3f}, {th[1,0]:.3f}, {th[2,0]:.3f}] | {gap_ne:.2e} |")
lines.append(f"\n- 真实参数 [5.000, 2.000, 3.000]；正规方程解是**这 100 个样本的最优估计**，"
             f"与真值的差距来自噪声（w2 标准误 ≈ {se[2]:.3f}），不是算法误差。")
lines.append(f"- 条件数 κ(XᵀX)：未缩放 {cond_raw:.1f} → 缩放后 {cond_scl:.2f}；"
             f"α 稳定上限 {1/lam_max_raw:.6f} → {1/lam_max_scl:.4f}。")
lines.append("- 结论：缩放不改问题，只改「Hessian 的条件数」；GD 迭代次数与 κ 成正比，"
             "而正规方程不迭代，所以**不需要缩放**（已用 1e-12 级一致性验证）。")

with open(os.path.join(SCRIPT_DIR, 'results.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
logging.info("对比结果表已保存为 results.md")


# ========== 6. 画图 ==========
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

# --- 左：收敛曲线（双对数轴，统一原始尺度）---
ax = axes[0]
ax.loglog(np.arange(1, IT_RAW + 1), losses_raw, color='crimson', lw=1.2, ls='--',
          label=f'无缩放 GD (α={LR_RAW:g})')
ax.loglog(np.arange(1, IT_SCL + 1), losses_scl, color='seagreen', lw=2.2,
          label=f'有缩放 GD (α={LR_SCL:g})')
ax.axhline(loss_ne, color='royalblue', ls=':', lw=2, label=f'正规方程 (1 步, {loss_ne:.3f})')
ax.axhline(J_star, color='gray', ls='-', lw=1, alpha=0.6, label='最优值 J*')
ax.set_xlabel('迭代次数（对数）')
ax.set_ylabel('MSE（原始尺度，对数）')
ax.set_title('收敛曲线：三法同一把尺子')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

# --- 中：达标所需轮数 ---
ax = axes[1]
names = [r[0] for r in results]
vals, colors, labels = [], [], []
for name, lr, budget, n, th, L in results:
    if n is None:
        vals.append(budget); labels.append(f'>{budget:,}'); colors.append('crimson')
    else:
        vals.append(n); labels.append(f'{n:,}'); colors.append('seagreen' if name == '有缩放 GD' else 'royalblue')
bars = ax.bar(names, vals, color=colors, alpha=0.85)
ax.set_yscale('log')
ax.set_ylabel('达标所需轮数（对数）')
ax.set_title(f'收敛速度：无缩放慢 {n_raw // n_scl} 倍')
for b, lb in zip(bars, labels):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.25, lb,
            ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# --- 右：参数对比 ---
ax = axes[2]
idx = np.arange(3); width = 0.2
series = [('真实参数', theta_true.reshape(-1, 1), 'black')] + \
         [(r[0], r[4], c) for r, c in zip(results, ['crimson', 'seagreen', 'royalblue'])]
for k, (nm, th, col) in enumerate(series):
    ax.bar(idx + (k - 1.5) * width, th.ravel(), width, label=nm, color=col, alpha=0.85)
ax.set_xticks(idx)
ax.set_xticklabels(['b', 'w1', 'w2'])
ax.set_ylabel('参数值')
ax.set_title('参数对比（均为原始尺度）')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'convergence_comparison.png'), dpi=150, bbox_inches='tight')
logging.info("图像已保存为 convergence_comparison.png")
logging.info("=" * 62)
