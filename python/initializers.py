'''
Initializer Function
'''
# dependency
import numpy as np
from math import sqrt
from utility import get_nparams

def glorot_uniform(input_size, output_size, bias=True, constraint=[0,1]):
    limit = sqrt(6 / (input_size + output_size))
    low = -limit if -limit > constraint[0] else constraint[0]
    high = limit if limit < constraint[1] else constraint[1]
    wmatrix = [np.random.uniform(low, high, 1) for u in range(input_size*output_size)]
    if bias:
        wmatrix.extend(np.random.uniform(low, high, 1) for b in range(output_size))
    return wmatrix

def random_uniform(input_size, output_size, bias=True, constraint=[0,1]):
    assert len(constraint) == 2
    wmatrix = [np.random.uniform(constraint[0], constraint[1], 1) for u in range(input_size*output_size)]
    if bias:
        wmatrix.extend(np.random.uniform(constraint[0], constraint[1], 1) for b in range(output_size))
    return wmatrix

# temporary not compatible
def random_normal(input_size, output_size, bias=True, m=-0, s=1):
    wmatrix = [np.random.normal(m, s, 1) for u in range(input_size*output_size)]
    if bias:
        wmatrix.extend(np.random.normal(m, s, 1) for b in range(output_size))
    return wmatrix