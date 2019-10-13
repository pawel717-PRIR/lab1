import pandas as pd
from sklearn import preprocessing

def standard_scaler(data):
    standard_scaler = preprocessing.StandardScaler()
    return standard_scaler.fit_transform(data)

def min_max_scaler(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)