#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/9 14:35
# @Author  : Aries
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# 保存、加载单位化算子的配置信息
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
print(scaler2.transform(test))
print(scaler2.transform(training_set))
