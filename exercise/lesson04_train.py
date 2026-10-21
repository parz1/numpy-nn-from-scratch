import numpy as np

# 打车记录：只有公里数和车费，不告诉网络计价规则
d = np.arange(0, 21, 1.0)
fare = 10 + 2 * np.maximum(0, d - 3) + 1 * np.maximum(0, d - 10)

# 缩放：都除以 10
s = d / 10
y = fare / 10
S = s.reshape(-1, 1)          # 21 行 × 1 列

# 瞎猜的起点
# w = np.array([0.0, 1.0, 1.0])
# b = np.array([1.0, -0.1, -0.5])
# v = np.array([1.0, 1.0, 1.0])
w = np.array([0.0, 1.0])
b = np.array([1.0, -0.1])
v = np.array([1.0, 1.0])
lr = 0.1

for step in range(5001):
    # 前向（forward）：跟 lesson04_taxi.py 一样
    z = S * w + b
    a = np.maximum(0, z)
    y_hat = a @ v
    L = ((y_hat - y) ** 2).mean()

    # 反向（backward）：跟 lesson02 的一样，只是变成了表
    dL_dyhat = 2 * (y_hat - y) / len(y)        # (21,)
    dv = a.T @ dL_dyhat                         # (3,)  每个神经元：所有行程的 a × dL_dyhat 加起来
    dL_da = dL_dyhat.reshape(-1, 1) * v         # (21,3)
    dz = dL_da * (z > 0)                        # (21,3) ReLU 关着的地方梯度为 0
    # lesson02 里是 dL_da*da_dz*x，这里 dz 已经包含前两项，再乘 S，然后往下加
    dw = (dz * S).sum(axis=0)
    # lesson02 里是 dL_da*da_dz，然后往下加
    db = dz.sum(axis=0)

    w = w - lr * dw
    b = b - lr * db
    v = v - lr * dv

    if step in (0, 100, 1000, 5000):
        print(step, "损失：", round(L, 6))

print("学到的拐点（公里）：", np.round(-b / w * 10, 2))
print("每段每公里加价（元）：", np.round(w * v, 2))
print("2/5/8/15/20 公里预测车费：", np.round(y_hat[[2, 5, 8, 15, 20]] * 10, 2))
