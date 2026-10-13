import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def forward(x, w, b, v):
    z = x * w + b
    a = ReLU(z)
    return (z, a, v * a)

def loss(y_hat, y):
    return (y_hat - y) * (y_hat - y)

def backward(x, y, z, a, y_hat, v):
    dL_dyhat = 2 * (y_hat - y)
    dL_da = dL_dyhat * v
    da_dz = np.asarray(z > 0, dtype=float).astype(float)
    dz_dw = x
    dz_db = 1
    dw = dL_da * da_dz * dz_dw
    db = dL_da * da_dz * dz_db
    dv = dL_dyhat * a
    return (dw, db, dv)
