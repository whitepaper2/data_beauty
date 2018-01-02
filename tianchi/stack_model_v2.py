#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 17:17
# @Author  : Aries
# @Site    : 
# @File    : stack_model_v2.py
# @Software: PyCharm


import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from scipy.stats import gmean
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from preprocess_data import *

def get_eta_parameter():
    eta_parameter = [0.09, 0.12, 0.14, 0.1, 0.24]
    return eta_parameter


def get_rf_estimator_num():
    estimators = [48, 40, 50, 44, 46]
    return estimators


def train_xgboost():
    today = get_today()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    # eta_parameter = get_eta_parameter()
    eta_parameter = [x/100 for x in range(1, 31)]
    eta_mse = np.zeros((len(eta_parameter), 2), dtype="float")
    for ind, eta in enumerate(eta_parameter):
        param = {'objective': 'reg:linear', 'silent': 1, 'eval_metric': ['error', 'rmse'], 'max_depth': 10, 'eta': eta}
        bst = xgb.train(param, dtrain, num_round, evallist)
        # 预测
        y_test_pred = bst.predict(dtest)
        mse = metrics.mean_squared_error(y_test, y_test_pred)
        eta_mse[ind] = np.array([eta, mse])
        print("eta={}, mse={}".format(eta, mse))
        joblib.dump(bst, "./output/{}/bstRegressor_model{}.m".format(today, eta))
    print(eta_mse)
    sort_by_mse = sorted(eta_mse.tolist(), key=lambda ele: ele[1])
    print(sort_by_mse)


def train_rf():
    today = get_today()
    x_train, x_test, y_train, y_test = get_train_test()
    scaler = StandardScaler()
    scaler.fit(x_train)  # Don't cheat - fit only on training data
    x_train_standard = scaler.transform(x_train)
    x_test_standard = scaler.transform(x_test)  # apply same transformation to test data
    # 3、单变量筛选
    print("step3: select topK cols...")
    selection = SelectKBest(mutual_info_regression, k=100)
    x_train_selection = selection.fit_transform(x_train_standard, y_train)
    x_test_selection = selection.transform(x_test_standard)
    selection_indices = selection.get_support(indices=True)  # 得到入模的列下标
    np.savetxt("./output/{}/mean_parameter.txt".format(today), scaler.mean_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/std_parameter.txt".format(today), scaler.var_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/selection_indices.txt".format(today), selection_indices, delimiter=',', fmt="%d")
    # 训练模型
    estimators = [x for x in range(10, 60, 2)]
    estimator_mse = np.zeros((len(estimators), 2), dtype="float")
    for ind, est in enumerate(estimators):
        clf = RandomForestRegressor(random_state=0, n_estimators=est, n_jobs=-1)
        clf.fit(x_train_selection, y_train)
        y_test_pred = clf.predict(x_test_selection)
        mse = metrics.mean_squared_error(y_test, y_test_pred)
        print("estimator_num={}, mse={}".format(est, mse))
        estimator_mse[ind] = np.array([est, mse])
        joblib.dump(clf, "./output/{}/rfRegressor_model{}.m".format(today, est))
    print(estimator_mse)
    sort_by_mse = sorted(estimator_mse.tolist(), key=lambda ele: ele[1])
    print(sort_by_mse)


def predict_newdata_byxgb():
    today = datetime.now().strftime("%Y%m%d")
    id, df = get_encode_newdata()
    xgb_data = xgb.DMatrix(df)
    [rows, cols] = df.shape
    # 加载xgboost模型
    eta_parameter = get_eta_parameter()
    pred_xgb = np.zeros((len(eta_parameter), rows), dtype="float")
    for ind, eta in enumerate(eta_parameter):
        bst = joblib.load("./output/{}/bstRegressor_model{}.m".format(today, eta))
        pred_xgb[ind] = bst.predict(xgb_data)
    # 据说几何平均数比算数平均数效果好
    # predict_avg = np.average(predictions, axis=0)
    predict_avg = gmean(pred_xgb, axis=0)
    result = list(zip(id, predict_avg))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")

if __name__ == "__main__":
    # initial()
    predict_newdata_byxgb()
    # train_xgboost()
    # train_rf()
    # test_model()
