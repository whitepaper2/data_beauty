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
from keras.layers import LSTM,Activation
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


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
    train = pdf[pdf["day_time"] <= 20200225]
    test = pdf[pdf["day_time"] > 20200225]
    print("train size={}".format(len(train)), ", test size={}".format(len(test)))
    feature = ["last2dcnt"]
    label = ["last1dcnt"]

    # print(pdf2.head())
    X_train = np.array(train.loc[:, feature])
    y_train = np.array(train.loc[:, label])
    X_test = np.array(test.loc[:, feature])
    y_test = np.array(test.loc[:, label])
    # sc = MinMaxScaler(feature_range=(0, 1))
    # X_train = sc.fit_transform(X_train2)
    # X_test = sc.transform(X_test2)
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

    regressor.add(LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))

    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))

    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))

    # regressor.add(LSTM(4))
    # regressor.add(Dropout(0.2))

    regressor.add(Dense(1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=5, batch_size=100)
    return regressor


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_preprocess()
    # sc = MinMaxScaler(feature_range=(0, 1))
    # y_train_scaled = sc.fit_transform(y_train)
    # y_test_scaled = sc.transform(y_test)
    # print(y_train_scaled)
    # print(y_test_scaled)

    model = build_model(X_train, y_train)
    print(model.summary())
    y_pred = model.predict(X_test)
    print(y_pred)
    # y_pred_scaled = sc.inverse_transform(y_pred)
    # print(y_pred_scaled)
    from sklearn.metrics import mean_squared_error
    import math
    rmse = math.sqrt(mean_squared_error(y_pred, y_test))
    print('Test RMSE: %.3f' % rmse)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
