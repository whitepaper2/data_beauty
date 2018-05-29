#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 上午8:57
# @Author  : pengyuan.li
# @Site    : 
# @File    : feature_engineer.py
# @Software: PyCharm

import prepro_data as pre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn import metrics
import xgboost as xgb


def get_dishist():
    x_train, x_test, y_train, y_test = pre.get_train_test()
    plt.figure()
    plt.hist(y_train)
    plt.savefig("./output/{}/y_train.png".format(pre.today))
    plt.close()
    print(y_train, type(y_train))


def get_single_feature():
    x_train, x_test, y_train, y_test = pre.get_train_test()
    df = pd.DataFrame(columns=('col_name', 'f1', 'recall', 'accuracy', 'auc_score'))

    for col in pre.get_select_cols():
        if col not in ["label"]:
            x_train_t = x_train[col].reshape(len(x_train[col]), 1)
            x_test_t = x_test[col].reshape(len(x_test[col]), 1)

            clf = xgb.XGBClassifier()
            clf.fit(x_train_t, y_train)

            y_pred = clf.predict(x_test_t)
            f1 = metrics.f1_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            auc_score = metrics.roc_auc_score(y_test, y_pred)
            s = pd.Series({"col_name": col, 'f1': f1, 'recall': recall, 'accuracy': accuracy, 'auc_score': auc_score})
            df = df.append(s, ignore_index=True)
            print("accuracy={}, recall={}, f1={}, auc={}".format(accuracy, recall, f1, auc_score))

            plt.figure()
            xgb.plot_tree(clf)
            plt.savefig("./output/tree/{}_tree.png".format(col), dpi=200)
            plt.close()
    df.to_csv("./output/{}/tree_df.csv".format(pre.today))


def get_feature_bin():
    x_train, x_test, y_train, y_test = pre.get_train_test()
    col = "f242"
    min = np.min(x_train[col])
    max = np.max(x_train[col])
    print(min, max)
    print("the number of null={}".format(x_train[col].isnull().sum() / len(x_train[col])))
    x_train[col] = x_train[col].apply(lambda x: 0.5 if np.isnan(x) else x)
    print("null ={} bin".format(1))
    bins = [min - 1, 1.5, 2.5, 3.5, 11.5, 103.5, max + 1]
    labels = [1, 2, 3, 4, 5, 6]
    x_train[col + "_bin"] = pd.cut(x_train[col], bins, right=True,
                                   labels=labels)
    # x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, max + 1], right=True,
    #                                labels=[1, 2, 3, 4])
    df = pd.concat([x_train[col + "_bin"], y_train], axis=1)
    total_rate = df.groupby([col + "_bin"])["label"].count() / len(df)
    bad_rate = df.groupby([col + "_bin"])["label"].sum() / df.groupby([col + "_bin"])["label"].count()
    # print(total_rate, bad_rate)
    rate = pd.concat([bad_rate, total_rate], axis=1)
    rate.columns = ["bad_rate", "total_rate"]
    print(rate)
    # plt.figure()
    # plt.plot(labels, total_rate, "bo-")
    # plt.plot(labels, bad_rate, "gv-")
    # # plt.bar(labels, rate)
    # plt.savefig("./output/feature_imgs/{}.png".format(col + "_bin"), dpi=200)
    # plt.close()


def get_all_features_bin():
    x_train, x_test, y_train, y_test = pre.get_train_test()
    col = "f7"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: 1.5 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, 4.5, 5.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5, 6])

    col = "f6"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 0.5, 1.5, 2.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])

    col = "f210"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: min - 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 2, min - 1, 3.5, 14.5, 23.5, 29.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5, 6])
    col = "f248"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 5.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])

    col = "f82"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: min - 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 2, min - 1, 79.95, 279.45, 587.2, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5])

    col = "f238"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f244"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f5"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: min - 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 2, min - 1, 100804, max + 1], right=True,
                                   labels=[1, 2, 3])
    col = "f236"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 4.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f237"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f243"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, max + 1], right=True,
                                   labels=[1, 2, 3])
    col = "f234"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 4, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f215"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, 6.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5])
    col = "f235"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4])
    col = "f218"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: max + 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, 4.5, 9.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5, 6])

    col = "f106"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: min - 1 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 2, min - 1, 6.5, 16.5, 27.5, 36.5, max + 1], right=True,
                                   labels=[1, 2, 3, 4, 5, 6])
    col = "f15"
    print("col={}".format(col))
    min, max = np.min(x_train[col]), np.max(x_train[col])
    print("min={}, max={}".format(min, max))
    x_train[col] = x_train[col].apply(lambda x: 0.5 if np.isnan(x) else x)
    x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 0.5, 1.5, max + 1], right=True,
                                   labels=[1, 2, 3])
    # col = "f4"
    # print("col={}".format(col))
    # min, max = np.min(x_train[col]), np.max(x_train[col])
    # x_train[col] = x_train[col].apply(lambda x: 0.5 if np.isnan(x) else x)
    # x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 0.5, 1.5, max + 1], right=True,
    #                                labels=[1, 2, 3])
    #
    # col = "f216"
    # print("col={}".format(col))
    # min, max = np.min(x_train[col]), np.max(x_train[col])
    # x_train[col] = x_train[col].apply(lambda x: 1.5 if np.isnan(x) else x)
    # x_train[col + "_bin"] = pd.cut(x_train[col], [min - 1, 1.5, 2.5, 3.5, 4.5, 6.5, max + 1], right=True,
    #                                labels=[1, 2, 3, 4, 5, 6])

    # x_train.to_csv("./output/{}/x_train_bins.csv".format(pre.today), index=None)


def feature_bin_pattern(df):
    col = "f7"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 1.5 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [- 1, 1.5, 2.5, 3.5, 4.5, 5.5, 8], right=True,
                              labels=[1, 2, 3, 4, 5, 6])

    col = "f6"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 5 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [- 1, 0.5, 1.5, 2.5, 5], right=True,
                              labels=[1, 2, 3, 4])

    col = "f210"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 0 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [-1, 0, 3.5, 14.5, 23.5, 29.5, 375], right=True,
                              labels=[1, 2, 3, 4, 5, 6])
    col = "f248"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 325 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 5.5, 325], right=True,
                              labels=[1, 2, 3, 4])

    col = "f82"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: - 1 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [- 2, - 1, 79.95, 279.45, 587.2, 688968.5], right=True,
                              labels=[1, 2, 3, 4, 5])

    col = "f238"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 677 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 677], right=True,
                              labels=[1, 2, 3, 4])
    col = "f244"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 229 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 229], right=True,
                              labels=[1, 2, 3, 4])
    col = "f5"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 9999 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [9998, 9999, 100804, 150407], right=True,
                              labels=[1, 2, 3])
    col = "f236"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 303 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 4.5, 303], right=True,
                              labels=[1, 2, 3, 4])
    col = "f237"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 368 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 368], right=True,
                              labels=[1, 2, 3, 4])
    col = "f243"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 280 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 280], right=True,
                              labels=[1, 2, 3])
    col = "f234"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 300 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 4, 300], right=True,
                              labels=[1, 2, 3, 4])
    col = "f215"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 300 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 6.5, 300], right=True,
                              labels=[1, 2, 3, 4, 5])
    col = "f235"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 303 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 303], right=True,
                              labels=[1, 2, 3, 4])
    col = "f218"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 303 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [0, 1.5, 2.5, 3.5, 4.5, 9.5, 303], right=True,
                              labels=[1, 2, 3, 4, 5, 6])

    col = "f106"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 0 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [-1, 0, 6.5, 16.5, 27.5, 36.5, 373], right=True,
                              labels=[1, 2, 3, 4, 5, 6])
    col = "f15"
    print("col={}".format(col))
    df[col] = df[col].apply(lambda x: 0.5 if np.isnan(x) else x)
    df[col + "_bin"] = pd.cut(df[col], [- 1, 0.5, 1.5, 3], right=True,
                              labels=[1, 2, 3])
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    select_cols = []
    for col in cat_columns:
        select_cols.append(col)

    for i in cat_columns:
        for j in cat_columns:
            if i != j:
                df["{}_{}".format(i, j)] = df[i].map(str) + df[j].map(str)
                select_cols.append("{}_{}".format(i, j))
            else:
                continue

    print(select_cols)
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder()
    # clf = enc.fit(df.loc[:, select_cols])
    # df2 = clf.transform(df.loc[:, select_cols])
    # print(df2)

    return df.loc[:, select_cols]


def get_onehot(df):
    enc = OneHotEncoder()
    clf = enc.fit(df)
    df2 = clf.transform(df)
    joblib.dump(clf, "./output/{}/onehot.pkl".format(pre.today))
    # print(df2)
    return df2


def onehot_newdata(df):
    clf = joblib.load("./output/{}/onehot.pkl".format(pre.today))
    df2 = clf.transform(df)
    return df2


if __name__ == "__main__":
    # get_single_feature()
    # get_feature_bin()
    # get_all_features_bin()
    x_train, x_test, y_train, y_test = pre.get_train_test()
    print(feature_bin_pattern(x_train))
