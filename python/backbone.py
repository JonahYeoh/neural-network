'''
Model
'''
# dependency
import numpy as np
import pandas as pd
import copy
import math
import sys
import os
from activations import relu, leaky_relu, sigmoid, tanh, softmax, d_sigmoid, d_relu
from initializers import glorot_uniform, random_normal, random_uniform
from regularizers import l1_regularizer, l2_regularizer
from utility import trim_tail, multiply, subtract, get_nparams, one_hot_encoding
from metrics import MSE, CCE, ACC, PRECISION, RECALL, essential_metrics, wondering_penalty, close_gap_penalty
from pso import PSO
from ga import GA
from nn import Network
from layer import DenseLayer
from scalers import MinMaxScaler, StandardScaler
'''

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
'''

metrics_fn_dict = dict()
metrics_fn_dict['accuracy'] = ACC
metrics_fn_dict['categorical_crossentropy'] = CCE
metrics_fn_dict['mean_square_error'] = MSE
metrics_fn_dict['precision'] = PRECISION
metrics_fn_dict['recall'] = RECALL
metrics_fn_dict['essential_metrics'] = essential_metrics
# n_weights = inputs size + 1


def get_data(root, dir, columns=None):
    with open(os.path.join(root, dir), 'r') as freader:
        lines = freader.readlines()
    records = list(map(lambda ele: list(map(lambda e: float(e), ele.split('\t'))), lines))
    df = pd.DataFrame(records)
    if columns is not None:
        df.columns = columns
    for col in df.columns:
        df[col] = df[col].astype('float32')
    print(df.info())
    return df

if __name__ == '__main__':
    model = Network(5, 0.001)
    model.add_layer(DenseLayer(1, 8, True, afn=sigmoid, dafn=None, rfn=None))#16
    model.add_layer(DenseLayer(2, 4, True, afn=sigmoid, dafn=None, rfn=None))#16
    model.add_layer(DenseLayer(3, 3, True, afn=softmax, dafn=None, rfn=None))#20
    ga = GA(
        dict(
            m = 200,
            x_rate = [0.3, 0.5],
            m_rate = [0.2, 0.6],
            weight_constraint = [-3.,3.],
            radioactive_rating = [0.1, 0.5],
            grow_factor = [0.1, 0.5]))

    model.compile('categorical_crossentropy', ga, ['accuracy', 'mean_square_error', 'essential_metrics', 'categorical_crossentropy'])
    DATA = os.path.join(os.getcwd(), 'dataset')
    GROUP = 2
    COLUMNS = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
    N_CLASS = 3
    EPOCHS = 300
    train_dir, test_dir = "..//dataset//training_data{}.txt".format(GROUP), "..//dataset//testing_data{}.txt".format(GROUP)
    train, test = get_data(DATA, train_dir, COLUMNS), get_data(DATA, test_dir, COLUMNS)
    # shuffle
    train, test = train.sample(frac=1), test.sample(frac=1)
    x_train, y_train = train.drop('label', axis=1), train['label']
    x_test, y_test = test.drop('label', axis=1), test['label']
    scaler = StandardScaler(['f1', 'f2', 'f3', 'f4', 'f5'])
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = one_hot_encoding(y_train, N_CLASS)
    y_test = one_hot_encoding(y_test, N_CLASS)
    model.fit(x_train, y_train, EPOCHS)
    score = model.evaluate(x_test, y_test, False)
    print('score', score)
    #print(model)