#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/18 下午2:06
# @Author  : pengyuan.li
# @Site    : 
# @File    : poc_generate_df.py
# @Software: PyCharm

import pandas as pd
from sklearn.utils import shuffle

path = "/Users/pengyuan.li/Documents/code/mycode/default_credit_Mac.csv"


def generate_df():
    # 3W, 48dim

    # df = pd.read_csv("/Users/pengyuan.li/Documents/code/mycode/add_feature_card.csv")
    df = pd.read_csv(path)
    df = shuffle(df)
    df_1w = df.iloc[1:10001, :]
    df_5w = df.append(df_1w).append(df_1w)
    df_10w = df_5w.append(df_5w)
    df_20w = df_10w.append(df_10w)
    df_50w = df_20w.append(df_20w).append(df_10w)
    df_100w = df_50w.append(df_50w)
    df_150w = df_100w.append(df_50w)
    df_200w = df_100w.append(df_100w)
    df_1000w = df_200w.append(df_200w).append(df_200w).append(df_200w).append(df_200w)

    # # print(df.head())
    # df_1w.to_csv("./add_df_1w.csv", index=None)
    # df_5w.to_csv("./add_df_5w.csv", index=None)
    #
    # df_10w.to_csv("./add_df_10w.csv", index=None)
    # df_20w.to_csv("./add_df_20w.csv", index=None)
    # df_50w.to_csv("./df_50w.csv", index=None)
    # df_100w.to_csv("./df_100w.csv", index=None)
    # df_150w.to_csv("./df_150w.csv", index=None)
    # df_200w.to_csv("./df_200w.csv", index=None)
    df_1000w.to_csv("./df_1000w.csv", index=None)
    # print(df_1w.shape, df_5w.shape, df_10w.shape, df_20w.shape, df_50w.shape, df_100w.shape)


def gen_df_by_num(num):
    df = pd.read_csv(path)
    df = shuffle(df)
    df_1w = df.iloc[1:10001, :]
    df = pd.DataFrame()
    for i in range(num):
        df = df.append(df_1w)
    print(df.shape)
    df.to_csv("./df_{}w.csv".format(num), index=None)


columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'V1', 'V2',
           'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
           'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
           'next_month']


def add_feature():
    df = pd.read_csv("/Users/pengyuan.li/Documents/code/mycode/信用卡还款数据集_Mac.csv")
    columns = ['BILL_AMT1',
               'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
               'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    for i in columns:
        columns.remove(i)
        for j in columns:
            if i != j:
                df["{}+{}".format(i, j)] = df[i] + df[j]
            else:
                continue
    # for i in columns:
    #     for j in columns:
    #         if i != j:
    #             df["{}-{}".format(i, j)] = df[i] - df[j]
    #         else:
    #             continue
    print(df.columns)
    df.to_csv("/Users/pengyuan.li/Documents/code/mycode/add_feature_card.csv", index=None)


if __name__ == "__main__":
    # add_feature()
    generate_df()
    # gen_df_by_num(1000)
