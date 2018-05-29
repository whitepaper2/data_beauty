#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 下午8:04
# @Author  : pengyuan.li
# @Site    : 
# @File    : main.py
# @Software: PyCharm

###
# python文件载入部分
###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

# verify_sample = pd.read_csv("verify_sample.csv")  # 加载验证数据集


###

###
# 将选手代码附于此处
###

def get_select_cols(is_train=True):
    labels = ['y']
    ID = ['user_id']
    cols = ['x_001', 'x_002', 'x_003', 'x_004', 'x_014', 'x_016', 'x_018', 'x_020', 'x_021', 'x_024',
            'x_025', 'x_026', 'x_027', 'x_029', 'x_030', 'x_031', 'x_032', 'x_033', 'x_034', 'x_035', 'x_036', 'x_041',
            'x_042', 'x_043', 'x_044', 'x_045', 'x_046', 'x_047', 'x_048', 'x_049', 'x_050', 'x_051', 'x_052', 'x_053',
            'x_054', 'x_055', 'x_056', 'x_057', 'x_058', 'x_059', 'x_060', 'x_061', 'x_065', 'x_066', 'x_067', 'x_074',
            'x_075', 'x_076', 'x_077', 'x_078', 'x_079', 'x_080', 'x_081', 'x_082', 'x_083', 'x_084', 'x_085', 'x_086',
            'x_087', 'x_088', 'x_089', 'x_090', 'x_091', 'x_092', 'x_093', 'x_094', 'x_095', 'x_096', 'x_097', 'x_098',
            'x_099', 'x_112', 'x_113', 'x_114', 'x_121', 'x_122', 'x_123', 'x_124', 'x_125', 'x_126', 'x_127', 'x_128',
            'x_129', 'x_130', 'x_131', 'x_132', 'x_133', 'x_134', 'x_135', 'x_136', 'x_137', 'x_138', 'x_139', 'x_140',
            'x_141', 'x_142', 'x_143', 'x_144', 'x_145', 'x_146', 'x_147', 'x_148', 'x_149', 'x_150', 'x_151', 'x_152',
            'x_153', 'x_154', 'x_155', 'x_156', 'x_157', 'x_158', 'x_159', 'x_160', 'x_161', 'x_162', 'x_163', 'x_164',
            'x_165', 'x_166', 'x_167', 'x_168', 'x_169', 'x_170', 'x_171', 'x_172', 'x_173', 'x_174', 'x_175', 'x_176',
            'x_177', 'x_178', 'x_179', 'x_180', 'x_181', 'x_182', 'x_183', 'x_184', 'x_185', 'x_186', 'x_187', 'x_188',
            'x_189', 'x_190', 'x_191', 'x_192', 'x_193', 'x_194', 'x_195', 'x_196', 'x_197', 'x_198', 'x_199']
    if is_train:
        res = labels + cols
    else:
        res = ID + cols
    return res


def get_train_test(df):
    # df = pd.read_csv("./model_sample.csv")
    rm_df = df[(True - df['user_id'].isin(['A10087', 'A02450', 'A17143']))]
    select_cols = get_select_cols()
    filter_df = rm_df.loc[:, select_cols]
    [rows, cols] = filter_df.shape
    y = filter_df.iloc[:, 0]
    x = filter_df.iloc[:, 1:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def fill_by_avg(df):
    x_train, x_test, y_train, y_test = get_train_test(df)
    from sklearn.preprocessing import Imputer

    scaler = Imputer().fit(x_train)
    transform_x_train = scaler.transform(x_train)
    transform_x_test = scaler.transform(x_test)
    scaler_path = './imputer.pkl'
    joblib.dump(scaler, scaler_path)
    return transform_x_train, transform_x_test, y_train, y_test


def train_by_gbdt(df):
    x_train, x_test, y_train, y_test = fill_by_avg(df)
    clf = GradientBoostingClassifier(max_depth=5, n_estimators=250,
                                     random_state=0)
    clf.fit(x_train, y_train)
    # y_test_pred = clf.predict(x_test)
    joblib.dump(clf, "./gbdt_model.m")


train_by_gbdt(df=model_sample)
select_cols = get_select_cols(False)
select_df = verify_sample.loc[:, select_cols]
[rows, cols] = select_df.shape
ID, x_df = select_df.iloc[:, 0], select_df.iloc[:, 1:cols]
scaler = joblib.load("./imputer.pkl")
transform_df = scaler.transform(x_df)
# 加载模型
clf = joblib.load("./gbdt_model.m")
pred = clf.predict(transform_df)
id_pred = list(zip(ID, pred))
predict_result = pd.DataFrame(id_pred)
predict_result.columns = ['user_id', 'y_prediction']


# ##
# python文件结束部分
# ##
f1_score = score(predict_result)  # 最终的f1值即参数选手的成绩
