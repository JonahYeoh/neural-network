'''
Model
'''
# dependency
import numpy as np
import pandas as pd
import copy
import math
import sys
from activations import relu, leaky_relu, sigmoid, tanh, softmax
from initializers import glorot_uniform, random_normal, random_uniform
from regularizers import l1_regularizer, l2_regularizer
from utility import trim_tail, multiply, subtract, get_nparams
from metrics import MSE, CCE, ACC, PRECISION, RECALL, essential_metrics, wondering_penalty, close_gap_penalty
from pso import PSO

activ_fn_dict = dict()
activ_fn_dict['relu'] = relu
activ_fn_dict['leaky_relu'] = leaky_relu
activ_fn_dict['sigmoid'] = sigmoid
activ_fn_dict['tanh'] = tanh

init_fn_dict = dict()
init_fn_dict['glorot_uniform'] = glorot_uniform
init_fn_dict['uniform'] = random_uniform
init_fn_dict['normal'] = random_normal

reg_fn_dict = dict()
reg_fn_dict['l1'] = l1_regularizer
reg_fn_dict['l2'] = l2_regularizer

loss_fn_dict = dict()
loss_fn_dict['categorical_crossentropy'] = CCE
loss_fn_dict['mean_square_error'] = MSE
loss_fn_dict['precision'] = PRECISION

metrics_fn_dict = dict()
metrics_fn_dict['accuracy'] = ACC
metrics_fn_dict['categorical_crossentropy'] = CCE
metrics_fn_dict['mean_square_error'] = MSE
metrics_fn_dict['precision'] = PRECISION
metrics_fn_dict['recall'] = RECALL
metrics_fn_dict['essential_metrics'] = essential_metrics

class ML_Agent(object):
    def __init__(self, n_feature, target_size=1):
        assert target_size > 0
        self.n_feature = n_feature
        self.problem = 'regression'
        if target_size > 0:
            self.problem = 'classification'
        self.target = target_size
        self.n_layer = 0
        self.units = []
        self.bias = []
        self.activation_functions = []
        self.init_functions = []
        self.regularizers = None
    
    def add_layer(self, units, activation=None, use_bias=True, init_fn='uniform'):
        self.n_layer += 1
        self.units.append(units)
        self.bias.append(use_bias)
        self.activation_functions.append(activ_fn_dict[activation] if activation is not None else False)
        self.init_functions.append(init_fn_dict[init_fn] if init_fn is not None else False)
        
    def compile_configuration(self, optimizer, loss='categorical_crossentropy', monitor=[], regularizer=None):
        #n_params = self.__build__() # redundant act
        self.n_params = get_nparams(self.n_feature, self.units, self.bias)
        self.optimizer = optimizer
        self.optimizer.__build__(self.n_params, self.n_feature, self.units, self.bias, self.init_functions)
        if regularizer is not None:
            self.regularizer = reg_fn_dict[regularizer]
        else:
            self.regularizer = regularizer
        self.loss_fn = loss_fn_dict[loss]
        self.monitor = dict()
        for item in monitor:
            self.monitor[item] = metrics_fn_dict[item]

    # won't be used
    def __build__(self):
        assert self.units[-1] == self.target
        self.weight_matrix = []
        self.bias_matrix = []
        prev = self.n_feature
        n_params = 0
        for layer in range(self.n_layer):
            current = self.units[layer]
            layer_weight = [np.absolute(np.random.uniform(0,0.05,1)) for u in range(current*prev)]
            self.weight_matrix.append(layer_weight)
            if self.bias[layer]:
                layer_bias = [np.absolute(np.random.uniform(0,0.05,1)) for b in range(current)]
                self.bias_matrix.append(layer_bias)
            else:
                self.bias_matrix.append(False)
            n_params += (current*prev) + (current if self.bias[layer] else 0)
            prev = current
        self.n_params = n_params
        return n_params
    
    def __load_weight__(self, weights):
        assert len(weights) == self.n_params, '{}:{}'.format(len(weights), self.n_params)
        self.flat_weight = copy.deepcopy(weights) # need enhancement wasting memory space
        self.weight_matrix = []
        self.bias_matrix = []
        prev = self.n_feature
        idx = 0
        for layer in range(self.n_layer):
            layer_weight = []
            layer_bias = []
            current = self.units[layer]
            for _ in range(current*prev):
                layer_weight.append(weights[idx])
                idx += 1
            if self.bias[layer]:
                for _ in range(current):
                    layer_bias.append(weights[idx])
                    idx += 1
            if self.bias[layer]:
                self.bias_matrix.append(layer_bias)
            else:
                self.bias_matrix.append(False)
            self.weight_matrix.append(layer_weight)
            prev = current
        #print('total params', idx)
        assert idx == len(weights)
    
    def fit(self, x, y, max_iter=100, goal=0.1, pack_goal=0.05, batch_size=5, validation=False, x_val=[], y_val=[], validation_batch_size=5):
        best_fitness, wmatrix = self.optimizer.fit(self, x, y, max_iter, goal, pack_goal, batch_size, validation, x_val, y_val, validation_batch_size)
        self.__load_weight__(wmatrix)
        return best_fitness
    
    def __forward__(self, x):
        #print('forward start', type(x), x)
        for idx in range(self.n_layer):
            #print('layer', idx+1)
            x = self.__get_output__(x, np.array(self.weight_matrix[idx]), np.array(self.bias_matrix[idx]))
            if not self.activation_functions[idx]:
                continue
            if idx != self.n_layer-1:
                activated = list(map(self.activation_functions[idx], x))
            else:
                activated = softmax(x)
            x = np.array(activated, dtype='float32')
        #print('forward end:', x.shape)
        return x
    
    def __get_output__(self, A, W, bias):
        # W, bias should be of the same type, ndarray
        assert type(W) == type(bias)
        in_size = len(A)
        out_size = len(W)//in_size
        assert out_size == len(W)/in_size
        #print(in_size, ':', out_size, 'bias len', len(bias))
        output = list()
        for unit in range(out_size):
            product = multiply(A, W[unit*in_size:(unit+1)*in_size])
            tmp = 0
            for a in range(in_size):
                tmp += np.sum(product[:,a])
            if type(bias) == list:
                tmp += bias[unit]
            output.append(tmp)
        return np.array(output)
    
    def _evaluate(self, x, y, batch_size=5):
        accumulated_loss = 0
        pred_y = list()
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].values.tolist()
            batch_y = y[i:i+batch_size].values.tolist()
            batch_pred = list()
            for step in range(len(batch_x)):
                xx = self.__forward__(batch_x[step])
                batch_pred.append(xx)
            accumulated_loss += self.__loss__(batch_pred, batch_y)
            pred_y.extend(batch_pred)
        if self.regularizer is not None:
            accumulated_loss = (accumulated_loss * 1) + (self.regularizer(self.flat_weight) * 1)
        return accumulated_loss, self.__monitor__(pred_y, y.values.tolist())
    
    def _predict(self, x, batch_size=5):
        pred_y = list()
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].values.tolist()
            batch_pred = list()
            for step in range(len(batch_x)):
                xx = self.__forward__(batch_x[step])
                if xx.shape != (self.target,1) and xx.shape == (self.target, ):
                    # print('fixing...', xx.shape, '-> ', (self.target, 1))
                    xx = np.reshape(xx, (3,1))
                if xx.shape[0] != self.target:
                    raise Exception('Unexpected Output Shape', 'Expected ({},1) but get {}'.format(self.target, xx.shape[0]))
                batch_pred.append(xx)
            pred_y.extend(batch_pred)
        output = np.array(pred_y, dtype='object')
        return trim_tail(output)
    
    def __loss__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true) + close_gap_penalty(y_pred)#+ wondering_penalty(y_pred) + close_gap_penalty(y_pred)

    def __monitor__(self, y_pred, y_true):
        metrics_board = dict()
        for key, fn in self.monitor.items():
            metrics_board[key] = fn(y_pred, y_true)
        return metrics_board