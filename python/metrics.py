'''
Metrics
'''
import numpy as np
import pandas as pd
import math
import sys
from utility import trim_tail, multiply, subtract
import tensorflow as tf
'''
Remarks: Need to ensure all y_pred and y_true are in the same dtype
'''
def MSE(y_pred, y_true):
    result = [l**2 for l in subtract(y_pred, y_true)]
    return np.sum(result) / len(y_pred)

def CCE(y_pred, y_true):
    '''
    if type(y_true) == pd.DataFrame:
        ground_truth = y_true.values.tolist()
    else:
        ground_truth = y_true
    '''
    total_CE = 0
    prob = []
    for idx in range(len(y_pred)):
        true_units = y_true[idx]
        pred_units = y_pred[idx]
        for i_target, i_pred in zip(true_units, pred_units):
            if i_target == 1:
                # required a proper smoothing operation
                i_pred = i_pred if i_pred > 1e-7 else 1e-7
                i_pred = i_pred if i_pred < 1 - 1e-7 else 1 - 1e-7
                prob.append(i_pred)

    prob = np.array(prob, dtype='float32')
    prob_tensor = tf.constant(prob)
    log_tensor = tf.math.log(prob_tensor)
    loss = tf.reduce_sum(log_tensor).numpy()

    total_CE = -1 * loss / len(y_pred)
    return total_CE

def ACC(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    hit = 0
    for idx in range(len(y_pred)):
        #print(np.argmax(batch_pred[idx]), 'vs', np.argmax(batch_y[idx]))
        if np.argmax(y_pred[idx]) == np.argmax(y_true[idx]):
            hit += 1
    accuracy = hit / len(y_pred)
    assert 0.0 <= accuracy <= 1.0, 'hit:{}, tot:{}'.format(hit, len(y_pred))
    return accuracy

def extremizer(array):
    extreme_array = list()
    for idx in range(len(array)):
        i_array = array[idx]
        max_idx = np.argmax(i_array)
        i_extreme_array = list()
        for jdx in range(len(i_array)):
            if jdx == max_idx:
                i_extreme_array.append(1)
            else:
                i_extreme_array.append(0)
        extreme_array.append(i_extreme_array)
    return np.array(extreme_array, dtype='float32')

def essential_metrics(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    book = dict()
    for element in range(len(y_pred[0])):
        book[element] = dict(tp_hit=0, fp_hit=0, tn_hit=0, fn_hit=0)
    y_pred = extremizer(y_pred)
    for idx in range(len(y_pred)):
        true_units = y_true[idx]
        pred_units = y_pred[idx]
        for cls, (i_target, i_pred) in enumerate(zip(true_units, pred_units), 0):
            if i_target == i_pred:
                if i_target == 0:
                    book[cls]['tn_hit'] = book[cls]['tn_hit'] + 1
                else:
                    book[cls]['tp_hit'] = book[cls]['tp_hit'] + 1
            elif i_target != i_pred:
                if i_target == 0:
                    book[cls]['fp_hit'] = book[cls]['fp_hit'] + 1
                else:
                    book[cls]['fn_hit'] = book[cls]['fn_hit'] + 1
    return book

def PRECISION(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    ref_book = essential_metrics(y_pred, y_true)
    global_tp = 0
    global_fp = 0
    for met in ref_book.values():
        global_tp += met['tp_hit']
        global_fp += met['fp_hit']
    return global_tp / (global_tp + global_fp)

def RECALL(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    ref_book = essential_metrics(y_pred, y_true)
    global_tp = 0
    global_fn = 0
    for met in ref_book.values():
        global_tp += met['tp_hit']
        global_fn += met['fn_hit']
    return global_tp / (global_tp + global_fn)

def wondering_penalty(y_pred):
    hit = 0
    for response in y_pred:
        if len(np.unique(response)) == 1:
            hit += 1
    #print('hit rate', hit/len(y_pred))
    return hit / len(y_pred)

def close_gap_penalty(y_pred, threshold=0.3):
    hit = 0
    for response in y_pred:
        if np.max(response) - np.min(response) < threshold:
            hit += 1
    #print('close gap hit rate', hit/len(y_pred))
    return hit / len(y_pred)
    
