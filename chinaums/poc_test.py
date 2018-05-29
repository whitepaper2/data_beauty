#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 下午2:37
# @Author  : pengyuan.li
# @Site    : 
# @File    : poc_test.py
# @Software: PyCharm

###
# python文件载入部分
###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
import xgboost as xgb
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
# import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# verify_sample = pd.read_csv("verify_sample.csv")  # 加载验证数据集
###

###
# 将选手代码附于此处
###

train_path = "./model_sample.csv"
test_path = "./test_sample.csv"


def get_select_cols(is_train=True):
    labels = ['y']
    ID = ['user_id']
    cols = ['x_001', 'x_002', 'x_003', 'x_004', "x_014", "x_020", "x_021", "x_131", "x_132", "x_147", "x_148"]
    if is_train:
        res = labels + cols
    else:
        res = ID + cols
    return res


def get_train_test():
    df = pd.read_csv(train_path)
    rm_df = df[(True - df['user_id'].isin(['A10087', 'A02450', 'A17143']))]
    select_cols = get_select_cols()
    filter_df = rm_df.loc[:, select_cols]
    # add_df = add_features(filter_df)
    [rows, cols] = filter_df.shape
    y = filter_df.iloc[:, 0]
    x = filter_df.iloc[:, 1:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test


def fill_by_avg():
    x_train, x_test, y_train, y_test = get_train_test()
    from sklearn.preprocessing import Imputer

    scaler = Imputer(strategy="median").fit(x_train)
    transform_x_train = scaler.transform(x_train)
    transform_x_test = scaler.transform(x_test)
    scaler_path = './imputer.pkl'
    joblib.dump(scaler, scaler_path)
    return transform_x_train, transform_x_test, y_train, y_test


def add_features(df):
    cols = ['x_021', 'x_024', 'x_025', 'x_026', 'x_029', 'x_030', 'x_031', 'x_032', 'x_034', 'x_035', 'x_036']
    for col in cols:
        df[col + '_1'] = df[col].apply(lambda x: 1 if x > 0 else 0)
    return df


def train_by_gbdt():
    x_train, x_test, y_train, y_test = fill_by_avg()
    clf = GradientBoostingClassifier(n_estimators=200, subsample=1.0, learning_rate=0.1, max_depth=3, random_state=0)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print("f1={}, accuracy={}, recall={}".format(f1, accuracy, recall))
    joblib.dump(clf, "./gbdt_model.pkl", protocol=0)


def train_by_xgboost():
    x_train, x_test, y_train, y_test = get_train_test()
    print(y_train)
    clf = xgb.XGBClassifier()
    # model = clf.fit()
    # y_test_pred = clf.predict(x_test)
    # f1 = f1_score(y_test, y_test_pred)
    # print("f1={}".format(f1))
    # joblib.dump(clf, "./xgb_model.pkl", protocol=0)
    pipeline = PMMLPipeline([('xgb', clf)])
    pipeline.fit(x_train, y_train)
    sklearn2pmml(pipeline, "./xgb3.pmml")
    # import matplotlib.pyplot as plt
    # plt.figure()
    # xgb.plot_tree(model, num_trees=10)
    # plt.savefig("./tree.png", dpi=200)
    # plt.close()
    # new_data = pd.DataFrame(
    #     {'x_001': [1], 'x_002': [2], 'x_003': [0], 'x_004': [3], "x_014": [4], "x_020": [5], "x_021": [1], "x_131": [2],
    #      "x_132": [22], "x_147": [3], "x_148": [5]})
    # print(new_data)
    # print(model.predict(new_data))


def predict_new_data():
    df = pd.read_csv("./test_sample.csv")

    select_cols = get_select_cols(False)
    select_df = df.loc[:, select_cols]
    [rows, cols] = select_df.shape
    ID, x_df = select_df.iloc[:, 0], select_df.iloc[:, 1:cols]
    scaler = joblib.load("./imputer.pkl")
    transform_df = scaler.transform(x_df)
    # 加载xgboost模型
    clf = joblib.load("./gbdt_model.pkl")
    pred = clf.predict(transform_df)
    result = list(zip(ID, pred))
    print(result)
    # np.savetxt("./test.csv", result, fmt="%s", delimiter=",")


def pipeline_test():

    # generate some data to play with

    df = pd.read_csv(train_path)
    [rows, cols] = df.shape
    from sklearn2pmml.pipeline import PMMLPipeline
    from sklearn2pmml import sklearn2pmml
    xgb_model = xgb.XGBClassifier()
    anova_svm = PMMLPipeline([('xgb', xgb_model)])
    clf = anova_svm.fit(df.iloc[:, 1:4], df.iloc[:, cols-1])


    sklearn2pmml(clf, "./xgb2.pmml", with_repr=True)
    # joblib.dump(clf, "./pipeline_model.pkl", protocol=0)
    model = joblib.load("./pipeline_model.pkl")
    print(hasattr(model, 'predict_proba'))
    # print(model.classes_)
    pred = pd.DataFrame(model.predict_proba(X[:, 1:4]))
    df = pd.DataFrame()
    df[["probability_0", "probability_1"]] = pred
    print(pred)
    print(df)


from sklearn.utils import shuffle
def generate_sample_df():
    cols = ['user_id','x_001', 'x_002', 'x_003', 'x_004', "x_014", "x_020", "x_021", "x_131", "x_132", "x_147", "x_148",'y']
    df = pd.read_csv(train_path)
    df = shuffle(df)
    print(df)
    df_20 = df.iloc[1:21, :].loc[:, cols]
    print(df_20)
    df_20.to_csv("./sample_20df.csv", index=None)

###
# python文件结束部分
###
# f1_score = score(predict_result)  # 最终的f1值即参数选手的成绩

if __name__ == "__main__":
    # get_data_summary(df)
    # train_by_gbdt()
    train_by_xgboost()
    # train_by_svm()
    # predict_new_data()
    # pipeline_test()
    # generate_sample_df()