#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 14:18
# @Author  : Aries
# @Site    : 
# @File    : stack_model_v3.py
# @Software: PyCharm

import lightgbm as lgbm
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from preprocess_data import *


def train_xgb_bygrid():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    base_estimator = xgb.XGBRegressor()
    parameters = [{'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200, 500, 1000],
                   'subsample': [0.75, 0.8, 0.85, 0.9], 'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]}]
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameters)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    joblib.dump(clf, "./output/{}/bstRegressor_model.m".format(today))


def train_xgb_mostpara():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    clf = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, reg_alpha=0.1, subsample=0.85)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    joblib.dump(clf, "./output/{}/xgbRegressor_model.m".format(today))


def train_rf_mostpara():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    clf = RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=100, max_features=0.9, min_samples_leaf=0.01)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    joblib.dump(clf, "./output/{}/rfRegressor_model2.m".format(today))


def train_rf_bygrid():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    base_estimator = RandomForestRegressor(n_jobs=-1)
    parameters = [{'max_depth': [5, 7, 10, 15], 'n_estimators': [100, 200, 500],
                   'max_features': [0.75, 0.8, 0.85, 0.9], 'min_samples_leaf': [0.01, 0.05, 0.1, 0.2]}]
    # parameters = [{'max_depth': [5, 7], 'n_estimators': [100, 200]}]
    clf = GridSearchCV(estimator=base_estimator, param_grid=parameters, verbose=1)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    # joblib.dump(clf, "./output/{}/rfRegressor_model.m".format(today))


def train_lgbm_bygrid():
    # x_train, x_test, y_train, y_test = get_minmax_train_test()
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    # base_estimator = lgbm.LGBMRegressor()
    # parameters = [
    #     {'num_leaves': [2**x for x in range(3, 10, 2)], 'n_estimators': [100, 200, 500, 1000], 'learning_rate': [0.01, 0.1, 0.3],
    #      'colsample_bytree': [0.75, 0.8, 0.85, 0.9], 'subsample': [0.75, 0.8, 0.85, 0.9],
    #      'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]}]
    # parameters = [{'max_depth': [5, 7], 'n_estimators': [100, 200]}]
    # clf = GridSearchCV(estimator=base_estimator, param_grid=parameters, verbose=1)
    clf = lgbm.LGBMRegressor()
    clf.fit(x_train, y_train)
    # print(clf.best_params_)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))


def train_lgbm_mostpara():
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    clf = lgbm.LGBMRegressor(colsample_bytree=0.85, learning_rate=0.1, n_estimators=200, num_leaves=32, reg_lambda=100, subsample=0.85)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    joblib.dump(clf, "./output/{}/lgbmRegressor_model.m".format(today))


def predict_newdata_byxgb():
    ID, en_df = get_encode_newdata()
    # 加载xgboost模型
    bst = joblib.load("./output/{}/bstRegressor_model.m".format(today))
    pred = bst.predict(en_df)
    result = list(zip(ID, pred))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")


def predict_newdata_byrf():
    ID, en_df = get_encode_newdata()
    # 加载xgboost模型
    bst = joblib.load("./output/{}/rfRegressor_model.m".format(today))
    pred = bst.predict(en_df)
    result = list(zip(ID, pred))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")


def predict_newdata_bylgbm():
    ID, en_df = get_encode_newdata()
    # 加载xgboost模型
    bst = joblib.load("./output/{}/lgbmRegressor_model.m".format(today))
    pred = bst.predict(en_df)
    result = list(zip(ID, pred))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")


def predict_newdata():
    # 加载xgboost模型
    # xgbr = joblib.load("./output/{}/xgbRegressor_model.m".format(today))
    # pred_xgb = xgbr.predict(en_df)
    ID, en_df = get_encode_newdata()
    rfr = joblib.load("./output/{}/bstRegressor_model.m".format(today))
    pred_rf1 = rfr.predict(en_df)
    print(pred_rf1)
    rfr = joblib.load("./output/{}/lgbmRegressor_model.m".format(today))
    pred_rf2 = rfr.predict(en_df)
    print(pred_rf2)
    pred = np.average(np.vstack((pred_rf1, pred_rf2)), axis=0, weights=[0.55, 0.45])
    result = list(zip(ID, pred))
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")


def test_model():
    # 加载xgboost模型
    # xgbr = joblib.load("./output/{}/xgbRegressor_model.m".format(today))
    # pred_xgb = xgbr.predict(en_df)
    x_train, x_test, y_train, y_test = get_train_test()
    rfr = joblib.load("./output/{}/bstRegressor_model.m".format(today))
    pred_rf1 = rfr.predict(x_test)
    print(pred_rf1)
    rfr = joblib.load("./output/{}/lgbmRegressor_model.m".format(today))
    pred_rf2 = rfr.predict(x_test)
    print(pred_rf2)
    pred = np.average(np.vstack((pred_rf1, pred_rf2)), axis=0, weights=[0.55, 0.45])
    mse = metrics.mean_squared_error(y_test, pred)
    print("mse={}".format(mse))


if __name__ == "__main__":
    start_time = datetime.now()
    # predict_newdata_byxgb()
    # train_xgboost()
    # train_rf()
    # test_model()
    # train_xgboost_bygrid()
    # train_xgboost_mostpara()
    # predict_newdata_mostpara()
    # train_rf_bygrid()
    # train_rf_mostpara()
    predict_newdata()
    # train_lgbm_bygrid()
    # train_lgbm_mostpara()
    # predict_newdata_bylgbm()
    finished_time = datetime.now()
    print("after {}s finish! ".format((finished_time - start_time).seconds))


