import numpy as np

def relu(x):
    if(x<0):
        return 0
    return x

def n1(x):
    return x - 26

def n2(x):
    return 18 - x

def y_hat(x):
    return relu(n1(x))+relu(n2(x))

print(y_hat(15),y_hat(22),y_hat(30))