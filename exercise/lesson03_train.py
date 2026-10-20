from lesson02_backprop import forward, loss, backward

# One sample
x, y = 2.0, 10.0
# Initialize
w, b, v = 1.0, 1.0, 2.0
# Learning Rate
lr = 0.02

for step in range(1, 11):
    z, a, y_hat = forward(x, w, b, v)
    L = loss(y_hat, y)
    dw, db, dv = backward(x, y, z, a, y_hat, v)

    w = w - lr * dw
    b = b - lr * db
    v = v - lr * dv

    print(step, "猜的是", y_hat, "损失", L)
