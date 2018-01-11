#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/4 10:20
# @Author  : Aries
# @Site    : 
# @File    : transform_data.py
# @Software: PyCharm
from preprocess_data import *
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def train_fill_data():
    df = get_train_data()
    ID, en_df = encode_onehot(df)
    imp = Imputer(strategy="mean")
    imp_train_test_x = imp.fit_transform(en_df.drop("Y", axis=1))
    imp_train_test = np.column_stack((en_df["Y"], imp_train_test_x))
    np.savetxt("./output/{}/imp_train_test.txt".format(today), imp_train_test)
    imp_stat = imp.statistics_
    np.savetxt("./output/{}/imp_stat.txt".format(today), imp_stat)


def fill_on_newdata():
    ID, en_df = get_encode_newdata()
    imp = Imputer(strategy="most_frequent")
    imp_stat = np.loadtxt("./output/{}/imp_stat.txt".format(today))
    imp.statistics_ = imp_stat
    imp_valid = imp.transform(en_df)
    # imp_valid = np.column_stack((ID, imp_df))
    np.savetxt("./output/{}/imp_id.txt".format(today), ID, fmt="%s", delimiter=",")
    np.savetxt("./output/{}/imp_valid.txt".format(today), imp_valid)


def train_minmax_scaler():
    imp_train_test = np.loadtxt("./output/{}/imp_train_test.txt".format(today))
    imp_train_test_x = imp_train_test[:, 1:]
    imp_train_test_y = imp_train_test[:, 0]
    scaler = MinMaxScaler()
    minmax_train_test_x = scaler.fit_transform(imp_train_test_x)
    minmax_train_test = np.column_stack((imp_train_test_y, minmax_train_test_x))
    np.savetxt("./output/{}/minmax_train_test.txt".format(today), minmax_train_test)
    np.savetxt("./output/{}/minmax_min.txt".format(today), scaler.min_)
    np.savetxt("./output/{}/minmax_scale.txt".format(today), scaler.scale_)


def minmax_on_newdata():
    imp_valid = np.loadtxt("./output/{}/imp_valid.txt".format(today))
    min2 = np.loadtxt("./output/{}/minmax_min.txt".format(today))
    scale = np.loadtxt("./output/{}/minmax_scale.txt".format(today))
    scaler = MinMaxScaler()
    scaler.scale_ = scale
    scaler.min_ = min2
    minmax_valid = scaler.transform(imp_valid)
    np.savetxt("./output/{}/minmax_valid.txt".format(today), minmax_valid)

if __name__ == "__main__":
    train_fill_data()
    fill_on_newdata()
    train_minmax_scaler()
    minmax_on_newdata()


