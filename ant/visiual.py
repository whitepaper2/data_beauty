#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 上午9:56
# @Author  : pengyuan.li
# @Site    : 
# @File    : visiual.py
# @Software: PyCharm
import prepro_data as pre
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')


def get_boxplot():
    df = pre.get_train_data()
    # df = df.convert_objects(convert_dates=False,convert_numeric=True,convert_timedeltas=False)
    fig, ax = plt.subplots(2, 2)
    plt.sca(ax[0, 0])
    plt.hist(df["300013"].dropna())
    plt.sca(ax[0, 1])
    # plt.boxplot(df["300013"].dropna())
    plt.hist(df["300013"].dropna().apply(lambda x: np.log(x) + 1))
    plt.sca(ax[1, 0])
    print(np.median(df["300013"].dropna()))
    print(np.max(df["300013"]))
    print(np.min(df["300013"]))
    plt.hist(df["300013"].replace(np.nan, np.median(df["300013"].dropna())))
    plt.sca(ax[1, 1])
    plt.hist(df["300013"].dropna().apply(lambda x: np.log(x + 1)))
    # plt.hist(df["300013"].replace(np.nan, np.mean(df["300013"].dropna())))

    plt.savefig("./output/imgs/{}.png".format("300013"), dpi=100)
    plt.close()
    # for col in df.columns:
    #     # plt.figure()
    #     fig, ax = plt.subplots(2, 2)
    #     plt.sca(ax[0, 0])
    #     # plt.boxplot(df[col].dropna())
    #     plt.hist(df[col].dropna())
    #     plt.sca(ax[0, 1])
    #     sns.distplot(df[col].dropna(), kde=True, rug=True, hist=True)
    #     plt.sca(ax[1, 0])
    #     plt.hist(df[col].replace(np.nan, np.mean(df[col])))
    #
    #     plt.savefig("./output/imgs/{}.png".format(col), dpi=100)
    #     plt.close()
    # plt.show()
    # print(df["300013"])


def get_violinplot():
    plt.figure()
    df = pre.get_train_data()
    # df = df.convert_objects(convert_dates=False,convert_numeric=True,convert_timedeltas=False)
    column_name = 'label1'
    sns.violinplot(df[column_name])
    # plt.violinplot()
    plt.savefig("./output/{}/violin_{}.png".format(pre.today, column_name), dpi=100)
    # plt.show()


def get_all_hist():
    df = pre.get_train_data(pre.round1_balanced_train_path)
    for col in df.columns:
        print(col)
        fig, ax = plt.subplots(2, 2)
        plt.sca(ax[0, 0])
        ax[0, 0].set_title(np.min(df[col].dropna()))
        plt.hist(df[col].dropna())
        plt.sca(ax[0, 1])
        ax[0, 1].set_title(np.max(df[col].dropna()))
        plt.hist(df[col].replace(np.nan, np.mean(df[col].dropna())))
        if np.min(df[col].dropna()) > 0:
            plt.sca(ax[1, 0])
            plt.hist(df[col].dropna().apply(lambda x: np.log(x) + 1))
            plt.sca(ax[1, 1])
            plt.hist(df[col].dropna().apply(lambda x: np.log(x + 1)))
        plt.savefig("./output/imgs/{}.png".format(col), dpi=100)
        plt.close()


if __name__ == "__main__":
    # get_boxplot()
    # get_violinplot()
    get_all_hist()
