#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/2 下午4:03
# @Author  : pengyuan.li
# @Site    : 
# @File    : 004_train_tcn.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from tcn import TCN

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##
milk = pd.read_csv('./datasets/monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)


def data_preprocess():
    """
    数据预处理，输出序列格式的数据
    :return:
    """
    pdf = pd.read_csv("./datasets/basic_route_hour_carcnt_label.csv")
    pdf = pdf.fillna(0)
    pdf = pdf[pdf["hour_time"] == 11]
    train = pdf[pdf["day_time"] <= 20200225]
    test = pdf[pdf["day_time"] > 20200225]
    print("train size={}".format(len(train)), ", test size={}".format(len(test)))
    feature = ["carflowcnt","last1dcnt", "last2dcnt", "last3dcnt", "last4dcnt", "last5dcnt", "last6dcnt", "last7dcnt", "last8dcnt",
               "last9dcnt", "last10dcnt", "last11dcnt",
               "last12dcnt", "last13dcnt", "last14dcnt"]
    label = ["next1dcnt"]

    X_train2 = np.array(train.loc[:, feature])
    y_train = np.array(train.loc[:, label])
    X_test2 = np.array(test.loc[:, feature])
    y_test = np.array(test.loc[:, label])
    sc = MinMaxScaler(feature_range=(0, 1))
    X_train = sc.fit_transform(X_train2)
    X_test = sc.transform(X_test2)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(X_train.shape, X_train.shape)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_preprocess()
    sc = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = sc.fit_transform(y_train)
    y_test_scaled = sc.transform(y_test)
    # print(y_train_scaled)
    # print(y_test_scaled)
    i = Input(shape=(15, 1))
    m = TCN()(i)

    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])

    model.summary()

    model.compile('adam', 'mae')

    print('Train...')
    model.fit(X_train, y_train_scaled, epochs=200, verbose=2)

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

# X_train, y_train, X_test, y_test = data_preprocess()
# print(milk.head())
#
# lookback_window = 12  # months.
#
# milk = milk.values  # just keep np array here for simplicity.
#
# x, y = [], []
# for i in range(lookback_window, len(milk)):
#     x.append(milk[i - lookback_window:i])
#     y.append(milk[i])
# x = np.array(x)
# y = np.array(y)
#
# print(x.shape)
# print(y.shape)
#
# i = Input(shape=(lookback_window, 1))
# m = TCN()(i)
# m = Dense(1, activation='linear')(m)
#
# model = Model(inputs=[i], outputs=[m])
#
# model.summary()
#
# model.compile('adam', 'mse')
#
# print('Train...')
# model.fit(x, y, epochs=100, verbose=2)
#
# p = model.predict(x)
#
# # plt.plot(p)
# # plt.plot(y)
# # plt.title('Monthly Milk Production (in pounds)')
# # plt.legend(['predicted', 'actual'])
# # plt.show()
