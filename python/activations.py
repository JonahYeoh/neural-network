'''
Activation Function
'''
# dependency
import numpy as np

def sigmoid(x):
    activated_list = list()
    try:
        for X in x:
            tmp = list()
            for Xi in X:
                if Xi == 0:
                    Xi = 1e-7
                y = 1 / (1 + np.exp(-Xi))
                tmp.append(y)
            activated_list.append(tmp)
        return activated_list
    except Exception as e:
        print('error msg', e, 'x', x)
        input('wait')
    return x

def d_sigmoid(x):
    s = sigmoid([x])[0][0].ravel()
    y = s * (1 - s)
    return y

def tanh(x):
    activated_list = list()
    for X in x:
        y = np.sinh(x) / np.cosh(x)
        activated_list.append(y)
    return activated_list

def relu(x):
    activated_list = list()
    for X in x:
        tmp = list()
        for Xi in X:
            y = 0 if Xi == 0 else Xi
            tmp.append(y)
        activated_list.append(tmp)
    return activated_list

def d_relu(x):
    return 0 if x == 0 else 1

def leaky_relu(x, alpha=0.01):
    activated_list = list()
    for X in x:
        y = X if X > 0 else alpha * X
        activated_list.append(y)
    return activated_list

def softmax(x):
    #print('soft in', x)
    soft_out = list()
    for X in x:
        activated_arr = list()
        if np.sum(X) == 0:
            soft_out.append(np.array([1/X.shape[0] for i in X], dtype='float32'))
            #print('here')
        else:
            activated_arr = list(map(lambda j: np.exp(j- np.max(X)), X))
            activated_sum = np.sum(activated_arr)
            for i in range(len(activated_arr)):
                activated_arr[i] /= activated_sum
                #print(activated_arr[i])
            #print('soft out', activated_arr)
            ret = np.array(activated_arr, dtype='float32')
            soft_out.append(ret)
    return soft_out