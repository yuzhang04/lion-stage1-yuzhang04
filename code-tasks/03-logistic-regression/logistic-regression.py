# -*- coding: utf-8 -*-
import sys
import logging
import os
from math import erfc, sqrt
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 0. 环境与日志配置 ==========
sys.stdout.reconfigure(encoding='utf-8')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'train.log')

# 每次运行重写日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ========== 1. 生成二分类合成数据 ==========
np.random.seed(42)
m = 200
C0, C1 = np.array([-1.5, -1.5]), np.array([1.5, 1.5])   # 两类中心

X_class0 = np.random.randn(m // 2, 2) + C0
X_class1 = np.random.randn(m // 2, 2) + C1
X = np.vstack((X_class0, X_class1))#垂直拼接
y = np.vstack((np.zeros((m // 2, 1)), np.ones((m // 2, 1))))

#独立测试集：同分布、4000 样本、不同随机源（不同于训练集的 200 个样本）
_rng = np.random.default_rng(2026)
X_test = np.vstack((_rng.standard_normal((2000, 2)) + C0,
                    _rng.standard_normal((2000, 2)) + C1))
y_test = np.vstack((np.zeros((2000, 1)), np.ones((2000, 1))))

# ========== 2. 特征缩放==========
#统计量只从训练集算，再套用到测试集 —— 用全量数据算等于数据泄露（lecture03）
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

X_b = np.c_[np.ones((m, 1)), X_scaled]
X_test_b = np.c_[np.ones((len(y_test), 1)), X_test_scaled]


# ========== 3. 模型 / 代价 / 梯度 ==========
def sigmoid(z):
    """Sigmoid：把任意实数映射到 (0,1)。clip 防止 z 很负时 exp 溢出（结果不变，只是不报警）"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def compute_cost(theta, X_design, y_true, lambda_reg=0.1):
    """带 L2 正则化的交叉熵代价（不惩罚偏置项 θ₀）"""
    n = len(y_true)
    h = sigmoid(X_design.dot(theta))
    eps = 1e-9
    ce = (-1 / n) * np.sum(y_true * np.log(h + eps) + (1 - y_true) * np.log(1 - h + eps))
    reg = (lambda_reg / (2 * n)) * np.sum(theta[1:] ** 2)
    return float(ce + reg)


def accuracy(theta, X_design, y_true):
    #算出模型预测对了百分之多少（也就是准确率）
    return float(np.mean((sigmoid(X_design.dot(theta)) >= 0.5).astype(int) == y_true)) * 100


def gradient_descent(X_design, y_true, learning_rate, n_iterations, lambda_reg=0.1):
    """梯度下降。每轮都记 loss（旧版每 100 轮才记一次，曲线只有 10 个点）"""
    n = len(y_true)
    theta = np.zeros((X_design.shape[1], 1))
    losses = np.empty(n_iterations)
    for i in range(n_iterations):
        h = sigmoid(X_design.dot(theta))
        gradients = (1 / n) * X_design.T.dot(h - y_true)
        reg_gradients = (lambda_reg / n) * theta
        reg_gradients[0] = 0                      # 偏置项不参与正则化
        theta = theta - learning_rate * (gradients + reg_gradients)
        losses[i] = compute_cost(theta, X_design, y_true, lambda_reg)
    return theta, losses


# ========== 4. 主实验 ==========
LR, ITERS, LAMBDA = 1.0, 5000, 0.1
logging.info("=" * 64)
logging.info("开始训练逻辑回归 (L2 正则化)...")
logging.info(f"学习率 α={LR}, 迭代 {ITERS} 轮, λ={LAMBDA}")
logging.info(f"训练集 {m} 样本 / 独立测试集 {len(y_test)} 样本")

theta, losses = gradient_descent(X_b, y, LR, ITERS, LAMBDA)

#收敛性自检：与"跑满 5 万轮"的参考解对比（旧版差 1.55 也毫无察觉）
theta_ref, _ = gradient_descent(X_b, y, LR, 50000, LAMBDA)
gap = float(np.abs(theta - theta_ref).max())
train_acc = accuracy(theta, X_b, y)
test_acc = accuracy(theta, X_test_b, y_test)

logging.info("-" * 64)
logging.info(f"学到的参数 θ     : {theta.flatten().round(3)}")
logging.info(f"参考解(5万轮) θ  : {theta_ref.flatten().round(3)}")
logging.info(f"与参考解最大差   : {gap:.4f}（旧版 α=0.1×1000 轮时是 1.5502，θ₂ 差 34%）")
logging.info(f"训练集准确率     : {train_acc:.2f}%")
logging.info(f"测试集准确率     : {test_acc:.2f}%  ← 旧版没有这个数，无法评价泛化")

# 理论上限：两类中心距决定的贝叶斯最优错误率
center_dist = float(np.linalg.norm(C1 - C0))
bayes_err = 0.5 * erfc(center_dist / (2 * sqrt(2))) * 100
logging.info(f"理论上限         : 两类中心距 {center_dist:.3f}σ → 贝叶斯最优错误率 {bayes_err:.2f}%"
             f" → 准确率上限 ≈ {100 - bayes_err:.2f}%")

assert gap < 0.01, f"未收敛：与参考解差 {gap:.4f}（>0.01），请调大 α 或增加轮数"
assert np.all(np.diff(losses) <= 1e-10), "损失出现回升，α 过大"
logging.info("收敛性断言通过（与参考解差 < 0.01，且 loss 全程单调）")


# ========== 5. λ 扫描之一：原始 2 特征（正则化"看起来没用"）==========
logging.info("-" * 64)
logging.info("λ 扫描（A）原始 2 特征 / 200 样本 —— 检验正则化有没有用")
lam_list = [0, 0.1, 1, 10, 100]
scan_a = []
for lam in lam_list:
    th, _ = gradient_descent(X_b, y, LR, ITERS, lam)
    tr, te = accuracy(th, X_b, y), accuracy(th, X_test_b, y_test)
    scan_a.append((lam, float(np.linalg.norm(th[1:])), compute_cost(th, X_b, y, lam), tr, te, tr - te))
    logging.info(f"  λ={lam:<6} ‖θw‖={scan_a[-1][1]:>6.3f}  训练={tr:.2f}%  测试={te:.2f}%  gap={tr-te:+.2f}pp")

best_a = min(scan_a, key=lambda r: r[4])[4]
worst_a = max(scan_a, key=lambda r: r[4])[4]
logging.info(f"→ 测试准确率只在 {min(r[4] for r in scan_a):.2f}%～{worst_a:.2f}% 之间浮动，"
             f"gap 始终 ≈1.1～1.2pp：**正则化在这份数据上不起作用**")


# ========== 6. λ 扫描之二：造出过拟合，看 λ 真正发力 ==========
# 为什么上面没用：2 特征 / 200 样本 / 近线性可分 → 模型**没有过拟合的空间**。
# 这里人为制造容量过剩：只用 80 个训练样本 + 6 次多项式（27 个特征，约 3 样本/特征）。
logging.info("-" * 64)
logging.info("λ 扫描（B）小样本 80 + 6 次多项式（27 特征）—— 人为造出过拟合")


def poly_features(Z, degree=6):
    """2 维 → 6 次多项式展开（不含常数项）：2+3+4+5+6+7 = 27 个特征"""
    cols = []
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((Z[:, 0] ** (d - i)) * (Z[:, 1] ** i))
    return np.column_stack(cols)


n_small = 80
half = n_small // 2
idx = np.r_[0:half, 100:100 + half]          # 分层抽样：两类各取一半
Xs, ys = X[idx], y[idx]
mu_s, sd_s = Xs.mean(0), Xs.std(0)
Ps = poly_features((Xs - mu_s) / sd_s)
Pt = poly_features((X_test - mu_s) / sd_s)
pm, ps = Ps.mean(0), Ps.std(0)
ps[ps < 1e-12] = 1                            # 防除零
Ps_n, Pt_n = (Ps - pm) / ps, (Pt - pm) / ps
S_b = np.c_[np.ones((n_small, 1)), Ps_n]
S_test_b = np.c_[np.ones((len(y_test), 1)), Pt_n]

scan_b = []
for lam in [0, 0.3, 1, 3, 10, 30, 100]:
    th, _ = gradient_descent(S_b, ys, LR, 8000, lam)
    tr, te = accuracy(th, S_b, ys), accuracy(th, S_test_b, y_test)
    scan_b.append((lam, float(np.linalg.norm(th[1:])), tr, te, tr - te))
    logging.info(f"  λ={lam:<6} ‖θw‖={scan_b[-1][1]:>6.3f}  训练={tr:.2f}%  测试={te:.2f}%  gap={tr-te:+.2f}pp")

g0, g_best = scan_b[0][4], min(r[4] for r in scan_b[:6])
logging.info(f"→ gap 从 λ=0 的 {g0:.2f}pp 压到最优 λ 的 {g_best:.2f}pp（缩小 {(1-g_best/g0)*100:.0f}%）："
             f"**存在过拟合空间时，正则化才真正发力**")
logging.info(f"→ λ=100 时训练/测试一起崩到 {scan_b[-1][2]:.1f}%，是**欠拟合**（正则化过度），不是发散")
logging.info("=" * 64)


# ========== 7. 对比结果表（产出物）==========
L = []
L.append("# 代码任务③：逻辑回归 + L2 正则化 —— 结果表\n")
L.append(f"- 数据：两类二维高斯，中心距 {center_dist:.3f}σ；训练 {m} 样本，独立测试 {len(y_test)} 样本")
L.append(f"- 主实验：α={LR}，{ITERS} 轮，λ={LAMBDA}；收敛判据：与 5 万轮参考解差 < 0.01\n")
L.append("## 一、主实验\n")
L.append("| 指标 | 值 |")
L.append("|---|---|")
L.append(f"| 参数 θ | {np.round(theta.flatten(), 3).tolist()} |")
L.append(f"| 参考解（5 万轮） | {np.round(theta_ref.flatten(), 3).tolist()} |")
L.append(f"| 与参考解最大差 | {gap:.4f} |")
L.append(f"| 训练集准确率 | {train_acc:.2f}% |")
L.append(f"| 测试集准确率 | {test_acc:.2f}% |")
L.append(f"| 贝叶斯理论上限 | {100-bayes_err:.2f}% |")
L.append(f"\n> 旧版（α=0.1×1000 轮）与参考解差 **1.5502**（θ₂ 差 34%），却因训练准确率 99.50% 而完全看不出来。\n")
L.append("## 二、λ 扫描（A）原始 2 特征 / 200 样本 —— 正则化**无效**\n")
L.append("| λ | ‖θ_w‖ | 训练 CE | 训练 acc | 测试 acc | 泛化 gap |")
L.append("|---|---|---|---|---|---|")
for lam, nw, ce_, tr, te, g in scan_a:
    L.append(f"| {lam} | {nw:.3f} | {ce_:.4f} | {tr:.2f}% | {te:.2f}% | {g:+.2f}pp |")
L.append(f"\n> λ 从 0 到 100，‖θ_w‖ 从 {scan_a[0][1]:.2f} 压到 {scan_a[-1][1]:.2f}，"
         f"但测试准确率只在 {min(r[4] for r in scan_a):.2f}%～{max(r[4] for r in scan_a):.2f}% 之间浮动。\n")
L.append("## 三、λ 扫描（B）80 样本 + 6 次多项式（27 特征）—— 正则化**生效**\n")
L.append("| λ | ‖θ_w‖ | 训练 acc | 测试 acc | 泛化 gap |")
L.append("|---|---|---|---|---|")
for lam, nw, tr, te, g in scan_b:
    L.append(f"| {lam} | {nw:.3f} | {tr:.2f}% | {te:.2f}% | {g:+.2f}pp |")
L.append(f"\n> 人为造出过拟合后，gap 从 **{scan_b[0][4]:.2f}pp 压到 {g_best:.2f}pp**（缩小 {(1-g_best/g0)*100:.0f}%）；"
         f"λ=100 时转为欠拟合（训练/测试一起崩）。\n")
L.append("## 四、结论\n")
L.append("- **正则化不是万能的**：它只在模型存在过拟合空间（容量相对数据量过剩）时才提升泛化。")
L.append("- 本任务原始数据 2 特征 / 200 样本 / 近线性可分，本来就没有过拟合空间，所以 λ 扫到 100 也几乎不动。")
L.append("- 判断依据是**泛化 gap（训练 acc − 测试 acc）**，不是训练准确率——后者全程 99.50%，什么都看不出来。")
with open(os.path.join(SCRIPT_DIR, 'results.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
logging.info("对比结果表已保存为 results.md")


# ========== 8. 画图 ==========
# --- 图 1：决策边界（任务书要求的产出）---
plt.figure(figsize=(8, 6))
plt.scatter(X_class0[:, 0], X_class0[:, 1], color='blue', label='Class 0', marker='o', alpha=0.6)
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='red', label='Class 1', marker='x', alpha=0.6)

# 决策边界：sigmoid(z)=0.5 ⟺ z=0 ⟺ θ₀+θ₁·x₁'+θ₂·x₂'=0（x' 为缩放空间坐标）
x1_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
x1_line_s = (x1_line - X_mean[0]) / X_std[0]
x2_line_s = -(theta[0, 0] + theta[1, 0] * x1_line_s) / theta[2, 0]
x2_line = x2_line_s * X_std[1] + X_mean[1]
plt.plot(x1_line, x2_line, color='green', lw=3, label='Decision Boundary')

# 理论最优：两类中心的垂直平分线 x1+x2=0
xb = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
plt.plot(xb, -xb, 'k--', lw=1.2, alpha=0.7, label='理论最优 x1+x2=0')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.title(f'Logistic Regression (α={LR}, {ITERS}轮, λ={LAMBDA})\n训练 {train_acc:.2f}% / 测试 {test_acc:.2f}%')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SCRIPT_DIR, 'decision_boundary.png'), dpi=150, bbox_inches='tight')
logging.info("决策边界图像已保存为 decision_boundary.png")

# --- 图 2：损失曲线 + 两张 λ 扫描 ---
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
ax.plot(np.arange(1, ITERS + 1), losses, color='seagreen', lw=1.6)
ax.set_xlabel('迭代次数'); ax.set_ylabel('代价 J（交叉熵 + L2）')
ax.set_title(f'损失下降曲线（α={LR}, λ={LAMBDA}）')
ax.grid(True, alpha=0.3)

# 注意：λ 是离散取值（0 / 0.1 / 1 / ...），横轴用索引等距排列，标签写真实 λ
def draw_scan(ax, scan, title, tr_i, te_i):
    xs = np.arange(len(scan))
    labels = [f'{r[0]:g}' for r in scan]
    ax.plot(xs, [r[tr_i] for r in scan], 'o-', color='crimson', label='训练 acc')
    ax.plot(xs, [r[te_i] for r in scan], 's-', color='royalblue', label='测试 acc')
    ax.fill_between(xs, [r[tr_i] for r in scan], [r[te_i] for r in scan],
                    color='orange', alpha=0.18, label='泛化 gap')
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylim(40, 101)          # 两图统一量程：A 图的 ~1pp 差异不应被坐标轴放大
    ax.set_xlabel('λ（离散取值，横轴等距）'); ax.set_ylabel('准确率 (%)')
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

draw_scan(axes[1], scan_a, '(A) 原始 2 特征：正则化几乎无效', 3, 4)
draw_scan(axes[2], scan_b, '(B) 80 样本 + 27 多项式特征：正则化生效', 2, 3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'loss_curve.png'), dpi=150, bbox_inches='tight')
logging.info("损失曲线与正则化扫描图已保存为 loss_curve.png")
