import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# 设置中文字体（Windows用 SimHei，Mac可以用 Songti SC）
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('../dateSet/priceBymultiLinear-dataset.csv')
X = data[['Size_sqm', 'floor', 'age_years', 'Distance_km']].values
y = data['Price_wan'].values.reshape(-1, 1)

# 数据标准化（梯度下降用）
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 添加一列 1 作为截距项
X_b = np.hstack([np.ones((X.shape[0], 1)), X])          # 正规方程用
X_scaled_b = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])  # 梯度下降用

# 方法1：正规方程
start_time = time.time()
theta_normal = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
time_normal = time.time() - start_time

# 方法2：梯度下降（指数衰减学习率）
def gradient_descent(X, y, lr0=0.05, decay=0.999, n_iter=5000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    lr_history = []
    for iteration in range(n_iter):
        lr = lr0 * (decay ** iteration)  # 指数衰减
        lr_history.append(lr)
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= lr * gradient
        cost = (1/(2*m)) * np.sum((X @ theta - y) ** 2)
        cost_history.append(cost)
    return theta, cost_history, lr_history

start_time = time.time()
theta_gd, cost_history, lr_history = gradient_descent(
    X_scaled_b, y, lr0=0.05, decay=0.999, n_iter=5000
)
time_gd = time.time() - start_time

# 输出对比
np.set_printoptions(precision=4, suppress=True)

print("=== 模型参数对比 ===")
print("正规方程 theta:", theta_normal.ravel())
print("梯度下降 theta:", theta_gd.ravel())

print("\n=== 用时对比 ===")
print(f"正规方程用时: {time_normal:.6f} 秒")
print(f"梯度下降用时: {time_gd:.6f} 秒")

# 图像展示
plt.figure(figsize=(12,5))

# 代价函数收敛曲线
plt.subplot(1,2,1)
plt.semilogy(range(len(cost_history)), cost_history)
plt.xlabel("迭代次数", fontsize=12)
plt.ylabel("代价函数 J(θ) (对数)", fontsize=12)
plt.title("梯度下降收敛过程", fontsize=14)
plt.grid(True)

# 学习率衰减曲线
plt.subplot(1,2,2)
plt.plot(range(len(lr_history)), lr_history)
plt.xlabel("迭代次数", fontsize=12)
plt.ylabel("学习率 α", fontsize=12)
plt.title("指数衰减学习率曲线", fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()
