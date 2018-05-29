#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 下午7:36
# @Author  : pengyuan.li
# @Site    : 
# @File    : train_model.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import prepro_data as pre
import feature_transform as feat
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split

# import lightgbm as lgb

warnings.filterwarnings("ignore")


def get_train_test():
    df = pre.get_train_data(pre.round1_balanced_train_path)
    # df2 = feat.feature_bin_pattern(df)

    [rows, cols] = df.shape
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)
    x_train_pre = feat.feature_bin_pattern(x_train)
    x_test_pre = feat.feature_bin_pattern(x_test)
    x_train_onehot = feat.get_onehot(x_train_pre)
    x_test_onehot = feat.onehot_newdata(x_test_pre)
    # x_test_onehot.toarray()
    # pd.DataFrame(x_test_onehot.toarray()).to_csv("./output/{}/x_test_onehot{}.csv".format(pre.today, pre.today_timestamp), index=None)

    return x_train_onehot, x_test_onehot, y_train, y_test

def plot_ks(preds, labels, n):
    # preds is score: asc=1
    # preds is prob: asc=0

    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad


    ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)


    ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.figure()
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
             color='blue', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
             color='red', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.ks, label='ks',
             color='green', linestyle='-', linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
    plt.xlabel('percentile')
    plt.ylabel('cum_rate%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.savefig("./output/{}/ks.png".format(pre.today))
    plt.close()

    return ksds

def train_by_xgb():
    # x_train, x_test, y_train, y_test = get_train_test()
    x_train, x_test, y_train, y_test = pre.get_train_test()
    print(x_train.shape)
    print(y_train.shape)
    clf = xgb.XGBClassifier(n_estimators=500, max_depth=3, random_state=2018, nthread=4, n_jobs=-1)
    # clf = LogisticRegression(max_iter=1000, n_jobs=-1, C=0.01, solver="lbfgs")
    # clf = Lasso(alpha=0.01)
    clf.fit(x_train, y_train)
    # Predict on new data
    y_pred = clf.predict(x_test)
    # print(clf.coef_)
    print(y_pred.shape)
    print(np.unique(clf.predict_proba(x_test)).shape)
    # np.savetxt("./output/{}/coef{}.csv".format(pre.today, pre.today_timestamp), clf.coef_, fmt="%s",
               # delimiter=",")
    # 评估

    f1 = metrics.f1_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    auc_score = metrics.roc_auc_score(y_test, y_pred)
    print("accuracy={}, recall={}, f1={}, auc={}".format(accuracy, recall, f1, auc_score))
    #
    # save model
    # joblib.dump(clf, "./output/{}/xgb.pkl".format(pre.today))

    # plot roc_auc
    y_score = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
    print(fpr, tpr, thresholds)

    result = list(zip(fpr,tpr,thresholds))
    np.savetxt("./output/{}/训练A.csv".format(pre.today), result, fmt="%s", delimiter=",")

    score = plot_ks(y_score, y_test, 100)
    score.to_csv("./output/{}/ks.csv".format(pre.today))
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.stackplot(fpr, tpr, color='darkorange')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig("./output/{}/auc_roc.png".format(pre.today), dpi=100)
    # plt.close()
    # feature_importances
    # names = np.array(x_train.columns)
    # feature_improtances = sorted(zip(names, clf.feature_importances_), key=lambda ele: ele[1])
    # print(feature_improtances)
    # pd.DataFrame(feature_improtances).to_csv("./output/{}/feature_importances{}.csv".format(pre.today, pre.today_timestamp), index=None)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # xgb.plot_tree(clf, num_trees=10)
    # plt.savefig("./tree.png", dpi=200)
    # plt.close()


def train_by_hyperopt():
    pass


def rfccv(n_estimators, max_depth, min_samples_split, max_features):
    X_train, X_test, y_train, y_test = pre.fill_by_avg()
    val = cross_val_score(
        RandomForestRegressor(n_estimators=int(n_estimators),
                              max_depth=int(max_depth),
                              min_samples_split=int(min_samples_split),
                              max_features=min(max_features, 0.999),
                              random_state=0,
                              n_jobs=-1
                              ),
        X_train, y_train, scoring='neg_mean_squared_error', cv=5
    ).mean()
    # return np.mean(np.sqrt(-val))
    return val


def train_by_bayesian():
    rfcBO = BayesianOptimization(
        rfccv,
        {'n_estimators': (10, 250),
         'max_depth': (25, 30),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999)}
    )
    print('-' * 53)
    rfcBO.maximize(n_iter=10)

    print('-' * 53)
    print('Final Results')
    print('RFC: %f' % rfcBO.res['max']['max_val'])


def predict_by_newdata():
    df = pre.get_newdata()
    [rows, cols] = df.shape
    ID = df.iloc[:, 0]
    x = df.iloc[:, 1:cols]
    # scaler_path = "./output/{}/imputer.pkl".format(pre.today)
    # scaler = joblib.load(scaler_path)
    model_path = "./output/{}/xgb.pkl".format(pre.today)
    model = joblib.load(model_path)
    # transform_df = scaler.transform(x)
    # pred = model.predict(x)
    pred = model.predict_proba(x)
    result = list(zip(ID, pred[:, 1]))
    result.insert(0, ("id", "score"))
    np.savetxt("./output/{}/round1_testa_{}.csv".format(pre.today, pre.today_timestamp), result, fmt="%s",
               delimiter=",")


def run_task(df, col):
    if col not in ["label"]:
        print(col)
        df[col + "_mean"] = df[col].apply(lambda x: x - np.mean(df[col].dropna()) if x is not None else x)
        df[col + "_abs"] = df[col].apply(lambda x: np.abs(x - np.mean(df[col].dropna())) if x is not None else x)


def multiprocess():
    df = pre.get_train_data(pre.round1_balanced_train_path)
    print(df.columns)
    import multiprocessing
    p = multiprocessing.Pool(processes=3)
    for col in df.columns:
        p.apply_async(run_task, args=(df, col,))
    p.close()
    p.join()
    df.to_csv("./output/{}/add_feature_df2.csv".format(pre.today), index=None)
    return df


if __name__ == "__main__":
    train_by_xgb()
    # predict_by_newdata()
    # train_by_bayesian()
    # multiprocess()
