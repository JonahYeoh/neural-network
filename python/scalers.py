'''
Scaler
'''
# dependency
import numpy as np
import pandas as pd
from scipy import stats
from functools import reduce
import os

class MinMaxScaler:
    def __init__(self, columns):
        self.columns = columns
        self._fitted = False

    def fit(self, df):
        self.f_diff = list()
        self.f_min = list()
        m = df.shape[1]
        mini_fn = lambda x, y: x if x <= y else y
        maxi_fn = lambda x, y: x if x >= y else y
        for c in self.columns:
            f_min = reduce(mini_fn, df[c])
            f_max = reduce(maxi_fn, df[c])
            self.f_min.append(f_min)
            self.f_diff.append(f_max-f_min)
        self._fitted = True

    def transform(self, data):
        if self._fitted == False:
            raise Exception('transform before scaler is fitted')
        tmp = pd.DataFrame()
        for i, c in enumerate(self.columns):
            tmp[c] = (data[c] - self.f_min[i]) / self.f_diff[i]
        assert tmp.shape == data.shape
        return tmp

class StandardScaler:
    def __init__(self, columns):
        self.columns = columns
        self._fitted = False
    
    def fit(self, df):
        self.f_mean = list()
        self.f_std = list()
        for col in self.columns:
            m = np.mean(df[col])
            s = np.std(df[col])
            self.f_mean.append(m)
            self.f_std.append(s)
        self._fitted = True

    def transform(self, data):
        if self._fitted == False:
            raise Exception('transform before scaler is fitted')
        tmp = pd.DataFrame()
        for i, col in enumerate(self.columns):
            tmp[col] = (data[col] - self.f_mean[i]) / self.f_std[i]
        assert tmp.shape == data.shape
        return tmp

def get_data(root, dir):
    with open(os.path.join(root, dir), 'r') as freader:
        lines = freader.readlines()
    records = list(map(lambda ele: list(map(lambda e: float(e), ele.split('\t'))), lines))
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    scaler = MinMaxScaler()
    DATA = os.path.join(os.getcwd(), 'dataset')
    GROUP = 1
    COLUMNS = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
    train_dir, test_dir = "..//dataset//training_data{}.txt".format(GROUP), "..//dataset//testing_data{}.txt".format(GROUP)
    train, test = get_data(DATA, train_dir), get_data(DATA, test_dir)
    train_f = train.drop(5, axis=1)
    train_l = train[5]
    print(train.shape, train_f.shape, train_l.shape)
    scaler.fit(train_f)
    scaled_f = scaler.transform(train_f)
    for s, r in zip(scaled_f[1], train_f[1]):
        print(s, 'vs', r)