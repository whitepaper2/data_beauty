#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 17:17
# @Author  : Aries
# @Site    : 
# @File    : stack_model_v2.py
# @Software: PyCharm


import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from preprocess_data import *


def get_eta_parameter():
    eta_parameter = [0.09, 0.12, 0.15, 0.16, 0.1, 0.24]
    return eta_parameter


def get_rf_estimator_num():
    estimators = [48, 40, 50, 44, 46]
    return estimators


def train_xgboost():
    today = get_today()
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    # eta_parameter = get_eta_parameter()
    eta_parameter = [x / 100 for x in range(1, 31)]
    eta_mse = np.zeros((len(eta_parameter), 2), dtype="float")
    for ind, eta in enumerate(eta_parameter):
        param = {'objective': 'reg:linear', 'silent': 1, 'eval_metric': ['error', 'rmse'], 'max_depth': 10,
                 'eta': eta, 'lambda': 0.5}
        bst = xgb.train(param, dtrain, num_round, evallist)
        # 预测
        y_test_pred = bst.predict(dtest)
        mse = metrics.mean_squared_error(y_test, y_test_pred)
        eta_mse[ind] = np.array([eta, mse])
        print("eta={}, mse={}".format(eta, mse))
        joblib.dump(bst, "./output/{}/bstRegressor_model{}.m".format(today, eta))
    print(eta_mse)
    sort_by_mse = sorted(eta_mse.tolist(), key=lambda ele: ele[1])
    print(sort_by_mse)


def train_xgboost_bygrid():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    base_estimator = xgb.XGBRegressor()
    parameters = [{'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200, 500, 1000],
                   'subsample': [0.75, 0.8, 0.85, 0.9], 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}]
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameters)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))


def train_xgboost_mostpara():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    clf = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, reg_alpha=0.1, subsample=0.85)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    joblib.dump(clf, "./output/{}/bstRegressor_model.m".format(today))


def train_rf():
    today = get_today()
    x_train, x_test, y_train, y_test = get_train_test()
    scaler = StandardScaler()
    scaler.fit(x_train)  # Don't cheat - fit only on training data
    x_train_standard = scaler.transform(x_train)
    x_test_standard = scaler.transform(x_test)  # apply same transformation to test data
    # 3、单变量筛选
    print("step3: select topK cols...")
    selection = SelectKBest(mutual_info_regression, k=100)
    x_train_selection = selection.fit_transform(x_train_standard, y_train)
    x_test_selection = selection.transform(x_test_standard)
    selection_indices = selection.get_support(indices=True)  # 得到入模的列下标
    np.savetxt("./output/{}/mean_parameter.txt".format(today), scaler.mean_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/std_parameter.txt".format(today), scaler.var_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/selection_indices.txt".format(today), selection_indices, delimiter=',', fmt="%d")
    # 训练模型
    estimators = [x for x in range(10, 60, 2)]
    estimator_mse = np.zeros((len(estimators), 2), dtype="float")
    for ind, est in enumerate(estimators):
        clf = RandomForestRegressor(random_state=0, n_estimators=est, n_jobs=-1)
        clf.fit(x_train_selection, y_train)
        y_test_pred = clf.predict(x_test_selection)
        mse = metrics.mean_squared_error(y_test, y_test_pred)
        print("estimator_num={}, mse={}".format(est, mse))
        estimator_mse[ind] = np.array([est, mse])
        joblib.dump(clf, "./output/{}/rfRegressor_model{}.m".format(today, est))
    print(estimator_mse)
    sort_by_mse = sorted(estimator_mse.tolist(), key=lambda ele: ele[1])
    print(sort_by_mse)


def train_rf_bygrid():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    base_estimator = RandomForestRegressor(n_jobs=-1)
    parameters = [{'max_depth': [5, 7, 10, 15], 'n_estimators': [100, 200, 500, 1000],
                   'max_features': [0.75, 0.8, 0.85, 0.9], 'min_samples_leaf': [0.01, 0.05, 0.1, 0.2]}]
    # parameters = [{'max_depth': [5, 7], 'n_estimators': [100, 200]}]
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameters, verbose=1)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))

def predict_newdata_byxgb():
    today = datetime.now().strftime("%Y%m%d")
    # id, df = get_encode_newdata()
    ID, en_df = get_encode_newdata()
    df = get_minmax_valid()
    xgb_data = xgb.DMatrix(df)
    [rows, cols] = df.shape
    # 加载xgboost模型
    eta_parameter = get_eta_parameter()
    pred_xgb = np.zeros((len(eta_parameter), rows), dtype="float")
    for ind, eta in enumerate(eta_parameter):
        bst = joblib.load("./output/{}/bstRegressor_model{}.m".format(today, eta))
        pred_xgb[ind] = bst.predict(xgb_data)
    # 据说几何平均数比算数平均数效果好
    predict_avg = np.average(pred_xgb, axis=0)
    # predict_avg = gmean(pred_xgb, axis=0)
    result = list(zip(ID, predict_avg))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")

def predict_newdata_mostpara():
    ID, en_df = get_encode_newdata()
    # 加载xgboost模型
    bst = joblib.load("./output/{}/bstRegressor_model.m".format(today))
    pred = bst.predict(en_df)
    result = list(zip(ID, pred))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")


if __name__ == "__main__":
    start_time = datetime.now()
    # initial()
    # predict_newdata_byxgb()
    # train_xgboost()
    # train_rf()
    # test_model()
    # train_xgboost_bygrid()
    # train_xgboost_mostpara()
    # predict_newdata_mostpara()
    train_rf_bygrid()
    finished_time = datetime.now()
    print("after {}s finish! ".format((finished_time-start_time).seconds))

