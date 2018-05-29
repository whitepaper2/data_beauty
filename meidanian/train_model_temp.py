#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 下午2:32
# @Author  : pengyuan.li
# @Site    : 
# @File    : train_model_temp.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.externals import joblib
import preprocess as pre
import warnings
import lightgbm as lgb

warnings.filterwarnings("ignore")


def train_by_multiout():
    X_train, X_test, y_train, y_test = pre.fill_by_avg()
    # X_train, X_test, y_train, y_test = pre.get_train_test()
    clf = MultiOutputRegressor(
        lgb.LGBMRegressor(max_depth=5, learning_rate=0.01, n_estimators=4000, reg_alpha=0, subsample=0.9, num_leaves=40,
                          n_jobs=-1, colsample_bytree=0.9), n_jobs=-1)
    clf.fit(X_train, y_train)
    # Predict on new data
    y_pred = clf.predict(X_test)
    # evaluation
    multi_mse = metrics.mean_squared_error(y_test, y_pred)
    multi_log_mse = metrics.mean_squared_log_error(y_test, y_pred)
    print("multi_mse={}, multi_log_mse={}".format(multi_mse, multi_log_mse))
    joblib.dump(clf, "./output/{}/multi_lgbm.pkl".format(pre.today))


def predict_by_newdata():
    # df = pre.get_newdata()
    print("step1: begin read table...")
    df = pd.read_csv("./output/20180426/add_test_df.csv")
    [rows, cols] = df.shape
    ID = df.iloc[:, 0]
    x = df.iloc[:, 1:cols]
    print("step2: transform data...")
    scaler_path = "./output/{}/imputer.pkl".format(pre.today)
    scaler = joblib.load(scaler_path)
    transform_df = scaler.transform(x)
    print("step3: predict...")
    model_path = "./output/{}/multi_lgbm.pkl".format(pre.today)
    model = joblib.load(model_path)
    pred = model.predict(transform_df)
    print("step4: save...")
    np.savetxt("./output/{}/round1_{}.csv".format(pre.today, pre.today_timestamp), pred, fmt="%s", delimiter=",")


if __name__ == "__main__":
    # train_by_multiout()
    predict_by_newdata()
