#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 下午2:01
# @Author  : pengyuan.li
# @Site    : 
# @File    : 001_train_lstm.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Activation
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


def data_analysis():
    pdf = pd.read_csv("./basic_route_hour_carcnt_label.csv")
    print(pdf.head())
    print(pdf.count())


def data_preprocess():
    """
    数据预处理，输出序列格式的数据
    :return:
    """
    pdf = pd.read_csv("./basic_route_hour_carcnt_label.csv")
    pdf = pdf.fillna(0)
    pdf = pdf[pdf["hour_time"] == 11]
    train = pdf[pdf["day_time"] <= 20200225]
    test = pdf[pdf["day_time"] > 20200225]
    print("train size={}".format(len(train)), ", test size={}".format(len(test)))
    feature = ["last1dcnt", "last2dcnt", "last3dcnt", "last4dcnt", "last5dcnt", "last6dcnt", "last7dcnt", "last8dcnt",
               "last9dcnt", "last10dcnt", "last11dcnt",
               "last12dcnt", "last13dcnt", "last14dcnt"]
    label = ["next1dcnt", "next2dcnt", "next3dcnt", "next4dcnt", "next5dcnt"]
    pdf2 = pdf[feature + label]

    # print(pdf2.head())
    X_train2 = np.array(train.loc[:, feature])
    y_train = np.array(train.loc[:, label])
    X_test2 = np.array(test.loc[:, feature])
    y_test = np.array(test.loc[:, label])
    sc = MinMaxScaler(feature_range=(0, 1))
    X_train = sc.fit_transform(X_train2)
    X_test = sc.transform(X_test2)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape[1], X_train.shape[2])
    return X_train, y_train, X_test, y_test


def build_model(X_train, y_train):
    """
    构建模型，
    :param X_train: 特征
    :param y_train: 标签
    :return: model
    """
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.5))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units=5))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=10, batch_size=64)
    return regressor


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_preprocess()
    sc = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = sc.fit_transform(y_train)
    y_test_scaled = sc.transform(y_test)
    # print(y_train_scaled)
    # print(y_test_scaled)

    model = build_model(X_train, y_train_scaled)
    print(model.summary())
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred_scaled = sc.inverse_transform(y_pred)
    print(y_pred_scaled)
    print(y_test)

    rmse = math.sqrt(mean_squared_error(y_pred_scaled, y_test))
    print('Test RMSE: %.3f' % rmse)

    # y_test2 = [x for y in y_test for x in y]
    # y_pred_scaled2 = [x for y in y_pred_scaled for x in y]
    # prediction = pd.DataFrame({"y_pred": y_pred_scaled, "y_test": y_test})
    # prediction.to_csv("./prediction.csv")
    err = 0
    cnt = 0
    err2 = 0
    for x, y in zip(y_pred_scaled, y_test):
        for x2, y2 in zip(x, y):
            if y2 != 0:
                err += abs(round(x2) - y2) / y2
            err2 += abs(round(x2) - y2) / (y2 + 0.1)
            cnt += 1
    print(err / cnt)
    print(err2 / cnt)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
