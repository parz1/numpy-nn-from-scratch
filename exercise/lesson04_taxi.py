import numpy as np

d = np.arange(0, 21, 1.0)                     # 0 到 20 公里，21 趟行程
# 真实车费，也对应着以往的程序怎么解决这个问题，用形式语言
fare = 10 + 2 * np.maximum(0, d - 3) + 1 * np.maximum(0, d - 10)
D = d.reshape(-1, 1)                          # 21 行 × 1 列

w = np.array([0.0, 1.0, 1.0])
b = np.array([10.0, -3.0, -10.0])
v = np.array([1.0, 2.0, 1.0])

# TODO：D 乘 w 再加 b，得到 21×3 的表
z = w * D + b
# ReLU
a = np.maximum(0, z)
# TODO：用 @ 把每一行合成一个车费
y_hat = a @ v

print("表的形状：", z.shape, a.shape, y_hat.shape)
print("2/5/15 公里的 a：\n", a[[2, 5, 15]])
print("2/5/15 公里的车费：", y_hat[[2, 5, 15]])
print("平均损失：", ((y_hat - fare) ** 2).mean())
