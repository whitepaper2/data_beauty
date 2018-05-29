#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/13 上午10:53
# @Author  : pengyuan.li
# @Site    : 
# @File    : test.py
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
from bayes_opt import BayesianOptimization
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# verify_sample = pd.read_csv("verify_sample.csv")  # 加载验证数据集
###

###
# 将选手代码附于此处
###

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
    op.to_csv("./data_summary.csv")


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


def get_train_test():
    df = pd.read_csv("./model_sample.csv")
    rm_df = df[(True - df['user_id'].isin(['A10087', 'A02450', 'A17143']))]
    select_cols = get_select_cols()
    filter_df = rm_df.loc[:, select_cols]
    add_df = add_features(filter_df)
    [rows, cols] = add_df.shape
    y = add_df.iloc[:, 0]
    x = add_df.iloc[:, 1:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test


def get_boxplot():
    plt.figure()
    df = pd.read_csv("./model_sample.csv")
    [rows, cols] = df.shape
    x = df.iloc[:, 2:cols - 1]
    y = df.iloc[:, 1]
    # for ind, r in enumerate(enc_x.columns):
    # plt.figure()
    # print(type(enc_x[r]))
    # print(dir(enc_x[r]))
    # print(enc_x[r])
    # plt.boxplot(enc_x[r])
    # plt.savefig("./output/imgs/{}.png".format(r), dpi=100)
    # plt.close()
    # plt.scatter(x['x_001'], y)
    plt.hist2d(x['x_001'], y)
    # plt.boxplot(x['x_001'])
    plt.savefig("./imgs/x_001.png", dpi=100)
    # plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend()
    plt.savefig("./imgs/learning_curve.png", dpi=100)


def test_learning_rate():
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = joblib.load("./gbdt_model.m")
    x_train, x_test, y_train, y_test = fill_by_avg()
    plot_learning_curve(estimator, "gbdt", x_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=4)


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
        df[col+'_1'] = df[col].apply(lambda x: 1 if x > 0 else 0)
    # df['x_0211'] = df['x_021'].apply(lambda x: 1 if x > 0 else 0)
    # df['x_0241'] = df['x_024'].apply(lambda x: 1 if x > 0 else 0)
    return df


def transform_by_minmax():
    x_train, x_test, y_train, y_test = get_train_test()
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x_train)
    transform_x_train = scaler.transform(x_train)
    transform_x_test = scaler.transform(x_test)
    scaler_path = './minmax.pkl'
    joblib.dump(scaler, scaler_path)
    return transform_x_train, transform_x_test, y_train, y_test


def train_by_gbdt():
    x_train, x_test, y_train, y_test = fill_by_avg()
    # parameters = {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [100, 200, 500, 1000],
    #               'subsample': [0.75, 0.8, 0.9, 1.0], 'max_depth': [3, 5, 7]}
    clf = GradientBoostingClassifier(n_estimators=1200, subsample=1.0, learning_rate=0.1, max_depth=3,
                                     random_state=0)
    # max_depth=5, n_estimators=250,
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    # base_estimator = GradientBoostingClassifier()
    # clf = RandomizedSearchCV(estimator=base_estimator, param_distributions=parameters, n_jobs=-1, n_iter=20)
    # clf = GridSearchCV(estimator=base_estimator, param_grid=parameters, n_jobs=-1)
    # clf.fit(x_train, y_train)
    # print(clf.best_params_)
    # y_test_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print("f1={}, accuracy={}, recall={}".format(f1, accuracy, recall))
    joblib.dump(clf, "./gbdt_model.m")


def train_by_xgboost():
    x_train, x_test, y_train, y_test = get_train_test()
    base_estimator = xgb.XGBClassifier()
    # parameters = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200, 500, 1000],
    #               'subsample': [0.75, 0.8, 0.85, 0.9], 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
    parameters = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 500, 1000]}
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameters, n_jobs=-1)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_test_pred = clf.predict(x_test)

    # rf = xgb.XGBClassifier(max_depth=max_depth, subsample=0.5, colsample_bytree=0.5, n_estimators=250)
    # rf.fit(x_train, y_train)
    # y_pred = rf.predict(x_test)
    f1 = f1_score(y_test, y_test_pred)
    print("f1={}".format(f1))


def predict_new_data():
    df = pd.read_csv("./test_sample.csv")

    select_cols = get_select_cols(False)
    select_df = df.loc[:, select_cols]
    [rows, cols] = select_df.shape
    ID, x_df = select_df.iloc[:, 0], select_df.iloc[:, 1:cols]
    scaler = joblib.load("./imputer.pkl")
    transform_df = scaler.transform(x_df)
    # 加载xgboost模型
    clf = joblib.load("./gbdt_model.m")
    pred = clf.predict(transform_df)
    result = list(zip(ID, pred))
    np.savetxt("./test.csv", result, fmt="%s", delimiter=",")


###
# python文件结束部分
###
# f1_score = score(predict_result)  # 最终的f1值即参数选手的成绩

if __name__ == "__main__":

    # get_data_summary(df)
    train_by_gbdt()
    # train_by_xgboost()
    # train_by_svm()
    # predict_new_data()
    # with open("./model_sample.csv") as f:
    #     for line in f.readlines():
    #         line_list = line.strip("\r\n").split(",")
    #         print(type(line_list))
    # get_boxplot()
    # test_learning_rate()
