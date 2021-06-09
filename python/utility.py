"""
Utility
"""
# dependency
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import os

def trim_tail(arr):
    result = arr
    if arr.shape[-1] == 1:
        tmp = []
        for row in arr:
            tmp.append(row.ravel())
        result = np.array(tmp)
    return result


def multiply(A, B):
    result = []
    for a in A:
        r = []
        for b in B:
            r.append(a * b)
        result.append(r)
    result = np.array(result)
    return trim_tail(result)


def subtract(A, B):
    assert len(A) == len(B)
    result = list()
    for a, b in zip(A, B):
        result.append(a - b)
    return result


def summation(array, decimal_point=4):
    total = 0
    for element in array:
        total += round(element, decimal_point)
    return total


def element_wise_multiply(a, b):
    assert len(a) == len(b)
    return [A * B for A, B in zip(a, b)]


def add(a, b):
    assert len(a) == len(b)
    return np.array([A + B for A, B in zip(a, b)], dtype="float32")


def devision(a, factor):
    assert factor != 0
    return np.array([ele / factor for ele in a], dtype="float32")

def one_hot_encoding(Target, n_classes):
    encoded_array = list()
    unique_label = np.unique(Target)
    rules = dict()
    for idx, label in enumerate(unique_label, 0):
        rules[label] = idx
    for target in Target:
        arr = np.zeros(n_classes)
        arr[rules[target]] = 1
        encoded_array.append(arr)
    return pd.DataFrame(np.array(encoded_array))

def split_data(dataframe, label, do_shuffle=True, test_size=0.1):
    if do_shuffle:
        dataframe = dataframe.sample(frac=1)
    unique_label = np.unique(dataframe[label])
    data_table = dict()
    for lbl in unique_label:
        data_table[lbl] = dataframe[dataframe[label] == lbl]
    train, validation, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for k, v in data_table.items():
        shuffled_v = v.sample(frac=1)
        trn = shuffled_v[: int(len(v) * 0.8)]
        val = shuffled_v[int(len(v) * 0.8) : int(len(v) * 0.9)]
        tst = shuffled_v[int(len(v) * 0.9) :]
        train = train.append(trn, ignore_index=True)
        validation = validation.append(val, ignore_index=True)
        test = test.append(tst, ignore_index=True)
    return train, validation, test

def split_data_2(dataframe, label, do_shuffle=True, test_size=0.1):
    if do_shuffle:
        dataframe = dataframe.sample(frac=1)
    unique_label = np.unique(dataframe[label])
    data_table = dict()
    for lbl in unique_label:
        data_table[lbl] = dataframe[dataframe[label] == lbl]
    train, test = pd.DataFrame(), pd.DataFrame()
    for k, v in data_table.items():
        shuffled_v = v.sample(frac=1)
        breakpoint = int(len(v) * 0.8)
        trn = shuffled_v[: breakpoint]
        tst = shuffled_v[breakpoint :]
        train = train.append(trn, ignore_index=True)
        test = test.append(tst, ignore_index=True)
    return train, test

def get_nparams(n_feature, units, bias):
    n_params = 0
    prev = n_feature
    for lyr, n_unit in enumerate(units):
        n_params += n_unit * prev + (n_unit if bias[lyr] else 0)
        prev = n_unit
    return n_params

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


if __name__ == "__main__":
    array = [np.random.uniform(0, 1, 1) for i in range(1000)]
    print(np.min(array), np.mean(array), np.max(array))
