'''
Regularizer
'''
# dependency
from functools import reduce

def l1_regularizer(w, l1=0.01):
    return l1 * reduce((lambda x, y: x+y), list(map((lambda x: abs(x)), w)))
    
def l2_regularizer(w, l2=0.01):
    return l2 * reduce((lambda x, y: x+y), list(map((lambda x: x**2), w)))
