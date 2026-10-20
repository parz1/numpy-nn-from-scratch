import numpy as np
from lesson02_backprop import forward, loss, backward

# 10℃ 到 34℃，25 个例子
T = np.arange(10, 35, 1.0)
# 正确答案：离舒适区差几度
y = np.maximum(0, T - 26) + np.maximum(0, 18 - T)
s = (T - 22) / 4

w1, b1, v1 = 1.0, 0.0, 1.0     # 神经元1：w 是正的，管"太热"
w2, b2, v2 = -1.0, 0.0, 1.0    # 神经元2：w 是负的，管"太冷"
lr = 0.02

for step in range(1, 1001):
    z1, a1, h1 = forward(s, w1, b1, v1)
    z2, a2, h2 = forward(s, w2, b2, v2)
    y_hat = h1 + h2
    L = loss(y_hat, y).mean()

    dw1, db1, dv1 = backward(s, y, z1, a1, y_hat, v1)
    dw2, db2, dv2 = backward(s, y, z2, a2, y_hat, v2)

    w1 = w1 - lr * dw1.mean()
    b1 = b1 - lr * db1.mean()
    v1 = v1 - lr * dv1.mean()

    w2 = w2 - lr * dw2.mean()
    b2 = b2 - lr * db2.mean()
    v2 = v2 - lr * dv2.mean()

    if step in (1, 100, 500, 1000):
        print(step, "平均损失", round(L, 5))

# 网络自己找到的舒适区边界：神经元刚好"开始开"的温度
print("太热从", 22 + 4 * (-b1 / w1), "度开始")
print("太冷从", 22 + 4 * (-b2 / w2), "度开始")

# 考考它
for t in (15, 22, 30):
    st = (t - 22) / 4
    print(t, "℃ 不舒服程度：", forward(st, w1, b1, v1)[2] + forward(st, w2, b2, v2)[2])
