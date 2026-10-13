import numpy as np

def relu(x):
    return np.maximum(0, x)

w1 = 1
b1 = -26
w2 = -1
b2 = 18
x = np.arange(10, 35, 4)

h1 = relu(w1 * x + b1)
h2 = relu(w2 * x + b2)

y_hat = h1 + h2

print(y_hat)
