'''
Activation Function
'''
# dependency
import numpy as np

def sigmoid(x):
    if x == 0:
        x = 1e-7
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.sinh(x) / np.cosh(x)

def relu(x):
    return leaky_relu(x, alpha=0.)

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def softmax(x):
    activated_arr = list()
    if np.sum(x) == 0:
        return np.array([1/x.shape[0] for i in x], dtype='float32')
    activated_arr = list(map(lambda j: np.exp(j), x))
    activated_sum = np.sum(activated_arr)
    for i in range(len(activated_arr)):
        activated_arr[i] /= activated_sum;
    ret = np.array(activated_arr, dtype='float32')
    assert x.shape[0] == ret.shape[0]
    return ret