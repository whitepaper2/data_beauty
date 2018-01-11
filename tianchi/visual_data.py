#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 14:33
# @Author  : Aries
# @Site    : 
# @File    : visual_data.py
# @Software: PyCharm

# 1、箱线图识别异常点
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocess_data as pre

# defaults
# plt.rcParams['figure.figsize'] = (200.0, 80.0)
# plt.rcParams.update({'font.size': 10})
# plt.rcParams['xtick.major.pad'] = '5'
# plt.rcParams['ytick.major.pad'] = '5'

plt.style.use('ggplot')

def get_boxplot():
    # plt.figure()
    df = pre.get_train_data()
    [rows, cols] = df.shape
    x = df.iloc[:, 0:cols - 1]
    y = df.iloc[:, cols - 1]
    id, enc_x = pre.encode_onehot(x)
    # enc_x.to_csv("./output/{}/en_df{}.csv".format(pre.today, pre.today_timestamp))
    for ind, r in enumerate(enc_x.columns):
        # plt.figure()
        print(type(enc_x[r]))
        print(dir(enc_x[r]))
        print(enc_x[r])
        # plt.boxplot(enc_x[r])
        # plt.savefig("./output/imgs/{}.png".format(r), dpi=100)
        # plt.close()
    # plt.boxplot(enc_x['210X1'])
    # plt.savefig("./output/imgs/test.png", dpi=100)
    # plt.show()


def get_split_boxplot():
    df = pre.get_train_data()
    reg_str, category_name, category_value = "^312X[0-9]+$", "Tool (#1)", [215, 329, 530, 1018, 1110, 1113, 1245, 2823]
    pattern = re.compile(reg_str)
    para = [x for x in df.columns if pattern.match(x)]
    for ind, r in enumerate(para):
        fig, ax = plt.subplots(2, 4)
        for i, tool in enumerate(category_value):
            row, col = int(i / 4), i % 4
            plt.sca(ax[row, col])
            ax[row, col].set_title(tool)
            filter_df = df[df[category_name].isin([tool])]
            # print(filter_df)
            filter_s = pd.Series(data=filter_df[r].values)
            plt.boxplot(filter_s, sym='o')
        # plt.show()
        plt.savefig("./output/split_imgs/{}.png".format(r))
        plt.close()
    # plt.boxplot(enc_x['210X1'])
    # plt.savefig("./output/imgs/test.png", dpi=100)
    # plt.show()


def modify_data():
    df = pre.get_train_data()
    df_copy = df.copy()
    # pattern = re.compile("^210X[0-9]+$")
    # para = [x for x in df.columns if pattern.match(x)]
    para = ['210X1', '210X2', '210X3', '210X4', '210X5']
    tool_id = ['J', 'K', 'L', 'M', 'N', 'O']
    # threshold = np.zeros((len(para)*len(tool_id), 4))
    threshold = list()
    for ind, r in enumerate(para):
        for i, tool in enumerate(tool_id):
            filter_df = df[df["TOOL_ID"].isin([tool])]
            filter_s = pd.Series(data=filter_df[r].values)
            min_threshold = filter_s.quantile(0.25)-1.5*(filter_s.quantile(0.75)-filter_s.quantile(0.25))
            max_threshold = filter_s.quantile(0.75)+1.5*(filter_s.quantile(0.75)-filter_s.quantile(0.25))

            # print(tool, r, min_threshold, max_threshold, filter_s.mean())
            # threshold.append((tool, r, min_threshold, max_threshold, filter_s.mean()))
            # print(df_copy[r])
            # print(filter_df.index)
            for ii in filter_df.index:
                if df_copy.loc[ii, r] > max_threshold or df_copy.loc[ii, r] < min_threshold:
                    df_copy.loc[ii, r] = filter_s.mean()
    df_copy.to_csv("./output/{}/df{}.csv".format(pre.today, pre.today_timestamp))


def modify_part_data(df, reg_str, category_name, category_value):
    df_copy = df.copy()
    pattern = re.compile(reg_str)
    cols = [x for x in df.columns if pattern.match(x)]
    threshold = list()
    for ind, r in enumerate(cols):
        for i, tool in enumerate(category_value):
            filter_df = df[df[category_name].isin([tool])]
            filter_s = pd.Series(data=filter_df[r].values)
            min_threshold = filter_s.quantile(0.25)-1.5*(filter_s.quantile(0.75)-filter_s.quantile(0.25))
            max_threshold = filter_s.quantile(0.75)+1.5*(filter_s.quantile(0.75)-filter_s.quantile(0.25))
            threshold.append((tool, r, min_threshold, max_threshold, filter_s.mean()))
            for ii in filter_df.index:
                if df_copy.loc[ii, r] > max_threshold or df_copy.loc[ii, r] < min_threshold:
                    df_copy.loc[ii, r] = filter_s.mean()
    # df_copy.to_csv("./output/{}/df{}.csv".format(pre.today, pre.today_timestamp))
    np.savetxt("./output/{}/threshold{}.csv".format(pre.today, pre.today_timestamp), threshold, fmt="%s")
    return df_copy


def modify_train_data():
    df = pre.get_train_data()
    reg_str, category_name, category_value = "^210X[0-9]+$", "TOOL_ID", ['J', 'K', 'L', 'M', 'N', 'O']
    df2 = modify_part_data(df, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^220X[0-9]+$", "Tool", ['A', 'B']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^300X[0-9]+$", "TOOL_ID (#1)", ['E', 'N']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^310X[0-9]+$", "TOOL_ID (#2)", ['E', 'C', 'D']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(311|261)X[0-9]+$", "TOOL_ID (#3)", ['E0', 'N0']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(312)X[0-9]+$", "Tool (#1)", [215, 329, 530, 1018, 1110, 1113, 1245, 2823]
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(330)X[0-9]+$", "Tool (#2)", ['A', 'B', 'C']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(340)X[0-9]+$", "tool", [3009, 4106, 4147]
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(344)X[0-9]+$", "tool (#1)", ['S', 'T', 'U', 'V', 'W', 'X']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(360|400|420)X[0-9]+$", "TOOL", ['B', 'C', 'D']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(520)X[0-9]+$", "Tool (#3)", [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    df_tmp = df2
    reg_str, category_name, category_value = "^(750)X[0-9]+$", "TOOL (#2)", ['B', 'A']
    df2 = modify_part_data(df_tmp, reg_str=reg_str, category_name=category_name, category_value=category_value)
    print(df2)
    df2.to_csv("./output/{}/modify_df{}.csv".format(pre.today, pre.today_timestamp))

def get_correlation():
    today = pre.get_today()
    df = pre.get_train_data()
    ID, enc_df = pre.encode_onehot(df)
    corrmat = enc_df.corr()
    corrmat.to_csv("./output/{}/corr.csv".format(today))


if __name__ == "__main__":
    # get_correlation()
    # get_boxplot()
    # get_split_boxplot()
    # modify_data()
    modify_train_data()
    # df.to_csv("./output/{}/df{}.csv".format(pre.today, pre.today_timestamp))
    # print(df.groupby('TOOL_ID')['210X1'])
    # print(df.groupby('TOOL_ID')['210X1'].mean())

