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
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import pandas as pd
import numpy as np
import os
import re


def get_today():
    today_time = datetime.now().strftime("%Y%m%d")
    out_path = "./output/{}".format(today_time)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return today_time


today_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

today = get_today()

data_part1_path = './dataset/meinian_round1_data_part1_20180408.txt'
data_part2_path = './dataset/meinian_round1_data_part2_20180408.txt'
_regex_bigint = re.compile(r'^[-+]?[0-9]+$')
_regex_double = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')


def read_data(path1, path2):
    data_dict = {}
    data_file1 = open(path1, "r", encoding="utf-8")
    data_text1 = data_file1.read()[1:].split("\n")[1:-1]
    data_file2 = open(path2, "r", encoding="utf-8")
    data_text2 = data_file2.read()[1:].split("\n")[1:-1]
    data_text1.extend(data_text2)
    for line in data_text1:
        line = line.split("$")
        if line[0] in data_dict:
            data_dict[line[0]][line[1]] = line[2]
        else:
            data_dict[line[0]] = {line[1]: line[2]}
    return pd.DataFrame(data_dict).T


def read_data1(path1):
    data_dict = {}
    data_file1 = open(path1, "r", encoding="utf-8")
    data_text1 = data_file1.read()[1:].split("\n")[1:-1]
    for line in data_text1:
        line = line.split("$")
        if line[0] in data_dict:
            data_dict[line[0]][line[1]] = line[2]
        else:
            data_dict[line[0]] = {line[1]: line[2]}
    return pd.DataFrame(data_dict).T


def merge_label_data():
    # train_data = read_data(data_part1_path, data_part2_path)
    # print(train_data.head())
    # train_data.to_csv("./dataset/meinian_round1_merge_20180408.csv")
    # train_data1 = read_data1(data_part2_path)
    # train_data1.to_csv("./dataset/meinian_round1_data2_20180409.csv", encoding='utf-8')
    # df = get_train_data()
    df = pd.read_csv("./output/20180411/data2_train_df.csv", encoding='utf-8')
    # df = pd.read_csv("./dataset/meinian_round1_merge_20180408.csv", encoding='utf-8')
    # label_data = pd.read_csv("./output/20180411/round1_train_df.csv", encoding='utf-8')
    # # # df.to_csv("./output/{}/data2_train_df.csv".format(today), encoding='utf-8', index=None)
    # label_train = pd.merge(label_data, df)
    # label_train.to_csv("./output/{}/train_df.csv".format(today), encoding='utf-8', index=None)

    # test_data = pd.read_csv("./dataset/[new] meinian_round1_test_a_20180409.csv", encoding='utf-8')
    test_data = pd.read_csv("./dataset/meinian_round1_test_b_20180505.csv", encoding="utf-8")
    merge = pd.merge(test_data, df)
    merge.to_csv("./output/{}/test_df.csv".format(today), encoding='utf-8', index=None)


def get_select_cols(is_train=True):
    labels = ['label1', 'label2', 'label3', 'label4', 'label5']
    ID = ['vid']
    cols = ['004997', '100005', '100007', '100012', '100013', '1106', '1107', '1115', '1117', '1125', '1127', '1345',
            '1844', '189', '269006', '269007', '269008', '269009', '269010', '269011', '269014', '269015', '269016',
            '269017', '269018', '269019', '269020', '269021', '269022', '269023', '269024', '269025', '279006', '2986',
            '300001', '300007', '30006', '300070', '300092', '31', '313', '315', '316', '317', '319', '33', '34', '37',
            '669001', '669004', '669005', '809004', '809007', '809008', '809009', '809010', '809013', '809017',
            '809018', '809019', '809020', '809022', '809023', '809026', '809027', '809029', '809031', '809032',
            '809033', '809034', '809037', '809038', '809039', '809040', '809041', '809042', '809043', '809044',
            '809045', '809046', '809047', '809048', '809049', '809050', '809051', '809053', '809054', '809055',
            '809056', '809057', '809058', '809059', '809060', '809061', '979010', '979024', '979025', '979026',
            '979027']
    if is_train:
        val = labels + cols
    else:
        val = ID + cols
    return val


def get_object_cols():
    cols = ['100006', '100014', '10002', '10003', '10004', '10009', '1110', '1112', '139', '143', '1474', '155', '1814',
            '1815', '183', '1840', '1845', '1850', '1873', '190', '191', '192', '193', '20002', '2174', '2333', '2371',
            '2372', '2386', '269003', '269004', '269005', '269012', '269013', '269026', '300008', '300009', '300011',
            '300012', '300013', '300014', '300021', '300035', '300068', '300074', '300076', '300078', '312',
            '314', '3193', '32', '320', '321', '669002', '669003', '669006', '669007',
            '669008', '669009', '669021', '809001', '809002', '809003', '809021', '809024', '809025', '809052',
            '979001', '979002', '979003', '979004', '979005', '979006', '979007', '979008', '979009', '979011',
            '979012', '979013', '979014', '979015', '979016', '979017', '979018', '979019', '979020', '979021',
            '979022', '979023']
    return cols


def get_train_data():
    # path = "./dataset/meinian_round1_data2_20180409.csv"
    path = "./output/20180411/train_df.csv"
    df = pd.read_csv(path, encoding='utf-8')
    # ignore_list = ['label1', 'label2', 'label3', 'label4', 'label5']
    # for col in ignore_list:
    #     df[col] = df[col].apply(lambda x: np.sqrt(x))
    # object_cols = get_object_cols()
    # transform_df = cols_autocast(df)
    select_cols = get_select_cols()
    return df.loc[:, select_cols]


def add_features(df):
    ignore_list = ['vid', 'label1', 'label2', 'label3', 'label4', 'label5']
    for col in df.columns:
        if col not in ignore_list:
            df[col + "_1"] = df[col].apply(lambda x: x - np.mean(df[col]))
            df[col + "_2"] = df[col].apply(lambda x: np.abs(x - np.mean(df[col])))
    return df


def get_train_test():
    df = get_train_data()
    # df = pd.read_csv("./output/20180426/add_train_df.csv")
    df = df[
        (df["label1"] < 225) & (df["label2"] < 130) & (df["label3"] < 6) & (df["label4"] < 3.5) & (df["label5"] < 7)]
    # df = df[(df["label3"] > 3) & (df["label3"] <= 4)]

    print(df.shape)
    # fill_df = fill_by_inf(df)
    [rows, cols] = df.shape
    y = df.iloc[:, 0:5]
    x = df.iloc[:, 5:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test


def get_train_test_label(label, i):
    # df["label3" + "_1"] = df["label3"].apply(lambda x: 1 if x > 1 else 0)
    # df["label3" + "_2"] = df["label3"].apply(lambda x: 1 if x > 2 else 0)
    # df["label3" + "_3"] = df["label3"].apply(lambda x: 1 if x > 3 else 0)
    # df["label3" + "_4"] = df["label3"].apply(lambda x: 1 if x > 4 else 0)
    df = get_train_data()
    # print(df.shape)
    df[label + i] = df[label].apply(lambda x: 1 if x > 4 else 0)
    [rows, cols] = df.shape
    y = df[label + i]
    x = df.iloc[:, 5:cols - 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test


def cols_autocast(df):
    cols = get_object_cols()
    for col in cols:
        df[col] = df[col].apply(
            lambda x: float(x) if _regex_bigint.match(str(x)) or _regex_double.match(str(x)) else np.NaN)
        # df.to_csv("./output/{}/data_transform{}.csv".format(today, today_timestamp))
        return df


def get_newdata():
    # path = "./output/20180411/test_df.csv"
    path = "./output/20180506/test_df.csv"
    df = pd.read_csv(path, encoding='utf-8')
    # print(df)
    object_cols = get_object_cols()
    select_cols = get_select_cols(False) + object_cols
    # for col in select_cols:
    #     df[col] = df[col].apply(lambda x: float(str(x)[:-1]) if str(x)[-1]=="." else x)
    transform_df = cols_autocast(df)
    return transform_df.loc[:, select_cols]


def fill_by_inf(df):
    ignore_list = ['vid', 'label1', 'label2', 'label3', 'label4', 'label5']
    for col in df.columns:
        if col not in ignore_list:
            df[col] = df[col].apply(lambda x: 999999 if np.isnan(x) else x)
    return df


def fill_by_avg():
    # x_train, x_test, y_train, y_test = get_train_test_label("label3", "_4")
    x_train, x_test, y_train, y_test = get_train_test()
    from sklearn.preprocessing import Imputer

    scaler = Imputer(strategy="median").fit(x_train)
    transform_x_train = scaler.transform(x_train)
    transform_x_test = scaler.transform(x_test)
    scaler_path = './output/{}/imputer.pkl'.format(today)
    joblib.dump(scaler, scaler_path)
    return transform_x_train, transform_x_test, y_train, y_test


def value_counts(das, nhead=5):
    tmp = pd.value_counts(das).reset_index().rename_axis({"index": das.name}, axis=1)
    value = pd.DataFrame(['value {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)).join(
        tmp.iloc[:, 0],
        how="left").set_index(
        0).T
    freq = pd.DataFrame(['freq {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)).join(
        tmp.iloc[:, 1],
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
    # x_train, x_test, y_train, y_test = fill_by_avg()
    # print(y_train)
    merge_label_data()
    # train = pd.read_csv("./output/20180425/train_df.csv", encoding='utf-8')
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

    # df = get_newdata()

    # part = df[(df["label5"] > 6) & (df["label1"] < 200)]
    # print(part.shape)
    # print(df.shape)
    # get_data_summary(df)
    # df2 = add_features(df)
    # df2.to_csv("./output/{}/add_test_df.csv".format(today), encoding='utf-8', index=None)
    # df.head()
    # get_train_test()
    # df = get_newdata()
    # df.to_csv("./output/{}/select_test_df.csv".format(today), encoding='utf-8', index=None)
