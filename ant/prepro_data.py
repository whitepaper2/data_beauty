#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 下午2:04
# @Author  : pengyuan.li
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import re

from dateutil.parser import parse


# df = get_train_data(round1_train_path)

def get_today():
    today_time = datetime.now().strftime("%Y%m%d")
    out_path = "./output/{}".format(today_time)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return today_time


today_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

today = get_today()

round1_train_path = './dataset/atec_anti_fraud_train.csv'
round1_testa_path = './dataset/atec_anti_fraud_test_a.csv'
round1_balanced_train_path = './output/20180427/balanced_train_df.csv'


def get_select_cols(is_train=True):
    labels = ['label']
    ID = ['id']
    # cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
    #         'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28',
    #         'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55',
    #         'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70',
    #         'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85',
    #         'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100',
    #         'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112', 'f113',
    #         'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125', 'f126',
    #         'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138', 'f139',
    #         'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151', 'f152',
    #         'f153', 'f154', 'f155', 'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164', 'f165',
    #         'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178',
    #         'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191',
    #         'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203', 'f204',
    #         'f205', 'f206', 'f207', 'f208', 'f209', 'f210', 'f211', 'f212', 'f213', 'f214', 'f215', 'f216', 'f217',
    #         'f218', 'f219', 'f220', 'f221', 'f222', 'f223', 'f224', 'f225', 'f226', 'f227', 'f228', 'f229', 'f230',
    #         'f231', 'f232', 'f233', 'f234', 'f235', 'f236', 'f237', 'f238', 'f239', 'f240', 'f241', 'f242', 'f243',
    #         'f244', 'f245', 'f246', 'f247', 'f248', 'f249', 'f250', 'f251', 'f252', 'f253', 'f254', 'f255', 'f256',
    #         'f257', 'f258', 'f259', 'f260', 'f261', 'f262', 'f263', 'f264', 'f265', 'f266', 'f267', 'f268', 'f269',
    #         'f270', 'f271', 'f272', 'f273', 'f274', 'f275', 'f276', 'f277', 'f278', 'f279', 'f280', 'f281', 'f282',
    #         'f283', 'f284', 'f285', 'f286', 'f287', 'f288', 'f289', 'f290', 'f291', 'f292', 'f293', 'f294', 'f295',
    #         'f296', 'f297']

    cols = ['f266', 'f12', 'f52', 'f208', 'f233', 'f249', 'f240', 'f241', 'f164', 'f184', 'f239', 'f81', 'f225', 'f53',
            'f216', 'f226', 'f251', 'f231', 'f86', 'f85', 'f222', 'f270', 'f31', 'f54', 'f263', 'f25', 'f26', 'f205',
            'f28', 'f17', 'f246', 'f247', 'f204', 'f217', 'f11', 'f209', 'f14', 'f29', 'f30', 'f18', 'f4', 'f245',
            'f253', 'f19', 'f242', 'f15', 'f106', 'f252', 'f218', 'f235', 'f215', 'f234', 'f243', 'f237', 'f236', 'f5',
            'f244', 'f238', 'f82', 'f248', 'f210', 'f6', 'f7']
    if is_train:
        val = labels + cols
    else:
        val = ID + cols
    return val


def get_train_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    df = shuffle(df)
    select_cols = get_select_cols()
    return df.loc[:, select_cols]


def get_train_test():
    df = get_train_data(round1_balanced_train_path)
    # print(df.columns)
    # for col in df.columns:
    #     if col not in ["label"]:
    #         print(col)
    #         df[col + "_mean"] = df[col].apply(lambda x: x - np.mean(df[col].dropna()) if x is not None else x)
    #         df[col + "_abs"] = df[col].apply(lambda x: np.abs(x - np.mean(df[col].dropna())) if x is not None else x)
    # df.to_csv("./output/{}/add_feature_df.csv".format(today))
    # df = pd.read_csv("./output/{}/add_feature_df.csv".format(today))
    [rows, cols] = df.shape
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)
    return x_train, x_test, y_train, y_test


def get_newdata():
    df = pd.read_csv(round1_testa_path, encoding='utf-8')
    select_cols = get_select_cols(False)
    return df.loc[:, select_cols]


def fill_by_avg():
    x_train, x_test, y_train, y_test = get_train_test()
    from sklearn.preprocessing import Imputer

    scaler = Imputer(strategy="median").fit(x_train)
    transform_x_train = scaler.transform(x_train)
    transform_x_test = scaler.transform(x_test)
    scaler_path = './output/{}/imputer.pkl'.format(today)
    joblib.dump(scaler, scaler_path)
    return transform_x_train, transform_x_test, y_train, y_test


def get_balanced_data():
    df = get_train_data(round1_train_path)
    df_sample1 = df[df["label"].isin(["1"])]
    df_sample0 = df[df["label"].isin(["0"])].sample(len(df_sample1))
    merge_df = pd.concat([df_sample0, df_sample1])
    return merge_df


def value_counts(das, nhead=5):
    tmp = pd.value_counts(das).reset_index().rename_axis({"index": das.name}, axis=1)
    value = pd.DataFrame(['value {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)).join(tmp.iloc[:, 0],
                                                                                                        how="left").set_index(
        0).T
    freq = pd.DataFrame(['freq {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)).join(tmp.iloc[:, 1],
                                                                                                      how="left").set_index(
        0).T
    nnull = das.isnull().sum()
    freqother = pd.DataFrame({das.name: [das.shape[0] - nnull - np.nansum(freq.values), nnull]},
                             index=["freq others", "freq NA"]).T
    op = pd.concat([value, freq, freqother], axis=1)
    return (op)


def get_data_summary(da):
    op = pd.concat([pd.DataFrame({"type": da.dtypes, "n": da.notnull().sum(axis=0)}), da.describe().T.iloc[:, 1:],
                    pd.concat(map(lambda i: value_counts(da.loc[:, i]), da.columns))], axis=1).loc[da.columns]
    op.index.name = "Columns"
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    op.to_csv("./output/{}/data_summary{}.csv".format(today, create_time))


if __name__ == "__main__":
    # df = get_train_data(round1_balanced_train_path)
    # df["f12"] = df["f12"].astype(object)
    # print(df.dtypes)
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder()
    # # data discretization
    # X = df["f12"]
    # enc.fit(X)
    # X = enc.transform(X)
    # print(X)
    # print(train['0101'].head())
    # print(train['0101'].value_counts())

    # cols_autocast(train)
    # for col in train.columns:
    #     print(train[col].value_counts())
    # train = pd.read_csv("./output/20180425/data_transform20180425111243.csv", encoding='utf-8')
    # get_data_summary(train)
    # print(train.dtypes)
    # train = train.convert_objects(convert_numeric=True)
    # print(train.dtypes)
    # cols = get_select_cols()
    # part = train.loc[:, cols]
    # print(type(part))
    # part.to_csv("./output/{}/select_df.csv".format(today), encoding='utf-8', index=None)
    # get_data_summary(train.loc[:, cols])
    # df.to_csv("./output/{}/df.csv".format(today))
    # ID, en_df = encode_onehot(df)
    # print(en_df.head())
    # en_df.to_csv("./output/{}/en_df.csv".format(today))
    # df = get_train_data()
    # get_data_summary(df)
    # from dateutil.parser import parse
    # df = get_train_data(round1_train_path)
    # print(df["datetime"].max, df["datetime"].min)
    # print(df.head())
    # print(df.info())
    # get_train_test()
    # df = get_newdata()

    # df = get_balanced_data()
    # df.to_csv("./output/{}/balanced_train_df.csv".format(today), encoding='utf-8', index=None)
    # print(df.head())
    # print(df.shape)
    rule = [
        {
            'file_name': 'algo_dc_ml_2c_gbdt',
            'model': "cf.GBTClassificationModel"
        },
        {
            'file_name': 'algo_dc_ml_2c_lr',
            'model': "cf.LogisticRegressionModel"
        }
    ]
    def get_model(rule, inputs0):
        for e in rule:
            print(e["file_name"])
            try:
                ind = inputs0.index(e["file_name"])
                # model = e['model'].load(inputs0)
                return e['model']
            except :
                continue
            # if inputs0.index(e['file_name']) > -1:
            #     model = e['model'].load(inputs0)
            #     return model
        raise Exception("not match model")
    print(get_model(rule, "hdfs://tdhdfs/user/turing/data/algo_dc_ml_2c_lr_364_1525834164887_356__0"))
