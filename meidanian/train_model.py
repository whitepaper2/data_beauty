#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 下午2:05
# @Author  : pengyuan.li
# @Site    : 
# @File    : train_model.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import preprocess as pre
import warnings
import xgboost as xgb

# import lightgbm as lgb

warnings.filterwarnings("ignore")


def train_by_multiout():
    X_train, X_test, y_train, y_test = pre.fill_by_avg()


    # X_train, X_test, y_train, y_test = pre.get_train_test()
    regr_multirf = MultiOutputRegressor(
        xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, reg_alpha=0.01, subsample=1, n_jobs=-1,
                         colsample_bytree=1), n_jobs=-1)
    regr_multirf.fit(X_train, y_train)
    # Predict on new data
    y_multirf = regr_multirf.predict(X_test)
    # 评估
    multi_mse = metrics.mean_squared_error(y_test, y_multirf)
    multi_log_mse = metrics.mean_squared_log_error(y_test, y_multirf)
    print("multi_mse={}, multi_log_mse={}".format(multi_mse, multi_log_mse))
    joblib.dump(regr_multirf, "./output/{}/multi_rf.pkl".format(pre.today))


def train_by_xgb():
    X_train, X_test, y_train, y_test = pre.fill_by_avg()

    scaler_path = "./output/20180503/reg_xgb.pkl"
    scaler = joblib.load(scaler_path)
    y1 = scaler.predict_proba(X_train)
    yt1 = scaler.predict_proba(X_test)

    scaler_path = "./output/20180503/reg_xgb2.pkl"
    scaler = joblib.load(scaler_path)
    y2 = scaler.predict_proba(X_train)
    yt2 = scaler.predict_proba(X_test)

    scaler_path = "./output/20180503/reg_xgb3.pkl"
    scaler = joblib.load(scaler_path)
    y3 = scaler.predict_proba(X_train)
    yt3 = scaler.predict_proba(X_test)

    scaler_path = "./output/20180503/reg_xgb4.pkl"
    scaler = joblib.load(scaler_path)
    y4 = scaler.predict_proba(X_train)
    yt4 = scaler.predict_proba(X_test)

    X_train_t = np.hstack((X_train, y1, y2, y3, y4)).astype(float)
    X_test_t = np.hstack((X_test, yt1, yt2, yt3, yt4)).astype(float)
    reg_xgb = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, reg_alpha=0.01, subsample=0.9,
                               colsample_bytree=0.9, n_jobs=-1, nthread=4)
    # print(y_train)
    reg_xgb.fit(X_train_t, y_train["label3"])
    print("Predict on new data...")
    y_pred = reg_xgb.predict(X_test_t)
    # # 评估
    multi_mse = metrics.mean_squared_error(y_test["label3"], y_pred)
    multi_log_mse = metrics.mean_squared_log_error(y_test["label3"], y_pred)
    print("multi_mse={}, multi_log_mse={}".format(multi_mse, multi_log_mse))
    joblib.dump(reg_xgb, "./output/{}/reg_xgbs.pkl".format(pre.today))


def train_by_xgb_pre():
    X_train, X_test, y_train, y_test = pre.fill_by_avg()
    # X_train, X_test, y_train, y_test = pre.get_train_test()
    reg_xgb = xgb.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=1000, reg_alpha=0.01, subsample=0.9,
                                colsample_bytree=0.9, n_jobs=-1)
    reg_xgb.fit(X_train, y_train["label3"])
    # Predict on new data
    y_pred = reg_xgb.predict(X_test)
    # 评估
    multi_mse = metrics.accuracy_score(y_test["label3"], y_pred)
    multi_log_mse = metrics.recall_score(y_test, y_pred)
    print("accuracy_score={}, recall_score={}".format(multi_mse, multi_log_mse))
    joblib.dump(reg_xgb, "./output/{}/reg_xgb4.pkl".format(pre.today))


def test_model():
    X_train, X_test, y_train, y_test = pre.fill_by_avg()
    model_path = "./output/20180428/multi_rf.pkl"
    model = joblib.load(model_path)
    y_multirf = model.predict(X_test)
    for i in range(0, 5):
        log_mse = metrics.mean_squared_log_error(np.power(y_test.iloc[:, i], 2), np.power(y_multirf[:, i], 2))
        print("log_mse{}:{}".format(i + 1, log_mse))


def predict_by_newdata():
    # df = pre.get_newdata()
    df = pd.read_csv("./output/20180506/test_df20180506143125.csv")
    [rows, cols] = df.shape
    ID = df.iloc[:, 0]
    x = df.iloc[:, 1:cols]
    # df.to_csv("./output/{}/test_df{}.csv".format(pre.today, pre.today_timestamp), encoding='utf-8', index=None)
    scaler_path = "./output/{}/imputer.pkl".format(pre.today)
    # scaler_path = "./output/20180425/imputer.pkl"
    scaler = joblib.load(scaler_path)
    model_path = "./output/{}/multi_rf.pkl".format(pre.today)
    # model_path = "./output/20180425/multi_rf.pkl"
    model = joblib.load(model_path)
    transform_df = scaler.transform(x)
    pred = model.predict(transform_df)
    # print(pred)
    # result = list(zip(ID.values, pred))
    #
    # print(result)
    np.savetxt("./output/{}/round1_{}.csv".format(pre.today, pre.today_timestamp), pred, fmt="%s", delimiter=",")


if __name__ == "__main__":
    # train_by_multiout()
    predict_by_newdata()
    # train_by_bayesian()
    # test_model()
    # train_by_xgb()
    # train_by_xgb_pre()
