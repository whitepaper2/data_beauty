#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 下午12:44
# @Author  : pengyuan.li
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]
log_error = mean_squared_log_error(y_true, y_pred)
print(log_error)

print(np.array(y_true) + 1)
print(np.log(np.array(y_true) + 1) - np.log(np.array(y_pred) + 1))
error_sum = np.sum(np.power(np.log(np.array(y_true) + 1) - np.log(np.array(y_pred) + 1), 2)) / 4
print(error_sum)

from datetime import datetime, date

print(datetime.now())
datetime.now().strftime("%Y")

training_set = np.random.rand(4, 4) * 10
scaler = MinMaxScaler()
scaler.fit(training_set)
tran_df = scaler.transform(training_set)
print(tran_df)
test = [[8.31263467, 7.99782295, 0.02031658, 9.43249727],
        [1.03761228, 9.53173021, 5.99539478, 4.81456067],
        [0.19715961, 5.97702519, 0.53347403, 5.58747666],
        [9.67505429, 2.76225253, 7.39944931, 8.46746594]]
model_path = './my_dope_model.pkl'
joblib.dump(scaler, model_path)

scaler2 = joblib.load(model_path)
print(dir(scaler2))
print(scaler2.transform(test))
print(scaler2.transform(training_set))
