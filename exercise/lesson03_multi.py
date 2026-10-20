import numpy as np
from lesson02_backprop import forward, loss, backward

x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 5.0, 7.0])
w, b, v = 1.0, 1.0, 1.0
lr = 0.01

for step in range(1, 301):
    z, a, y_hat = forward(x, w, b, v)
    L = loss(y_hat, y).mean()
    dw, db, dv = backward(x, y, z, a, y_hat, v)

    w = w - lr * dw.mean()
    b = b - lr * db.mean()
    v = v - lr * dv.mean()

    if step in (1, 2, 3, 10, 50, 100, 300):
        print(step, "猜的是", y_hat.round(3), "平均损失", round(L, 5))

print(forward(4.0, w, b, v)[2])
