'''
Scaler
'''
# dependency
import numpy as np
import pandas as pd
from scipy import stats

def StandardScaler(dataframe):
    scaled_dataframe = pd.DataFrame()
    for col in dataframe.columns:
        data_col = dataframe[col]
        scaled_dataframe[col] = stats.zscore(data_col)
    return scaled_dataframe

def MinMaxScaler(dataframe):
    scaled_dataframe = pd.DataFrame()
    for col in dataframe.columns:
        data_col = dataframe[col]
        #scaled_dataframe[col] = data_col / np.max(data_col)
        scaled_dataframe[col] = (data_col-np.min(data_col)) / (np.max(data_col)-np.min(data_col))
    return scaled_dataframe