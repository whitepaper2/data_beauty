from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from preprocess_data import *

# defaults
plt.rcParams['figure.figsize'] = (200.0, 80.0)
plt.rcParams.update({'font.size': 10})
plt.rcParams['xtick.major.pad'] = '5'
plt.rcParams['ytick.major.pad'] = '5'

plt.style.use('ggplot')

def process_data():
    path = "./dataset/训练.xlsx"
    df = pd.read_excel(path)
    # 1、删除文本格式、时间格式
    print("step1: delete some cols begin...")
    del_columns = get_del_cols()
    del_df = df.drop(del_columns, axis=1)
    # print(del_df.columns, "====>", del_df.head())
    # da = del_df
    # op = pd.concat([pd.DataFrame({"type": da.dtypes, "n": da.notnull().sum(axis=0)}), da.describe().T.iloc[:, 1:],
    #                 pd.concat(map(lambda i: Value_counts(da.loc[:, i]), da.columns))], axis=1).loc[da.columns]
    # op.index.name = "Columns"
    # op.to_csv("{}/{}_summary_da.csv".format("./output", "train3"))
    # 2、统一量纲
    print("step2: standard some cols...")
    [rows, cols] = del_df.shape
    x = del_df.iloc[:, 0:cols - 1]
    y = del_df.iloc[:, cols - 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x_train)  # Don't cheat - fit only on training data
    x_train_standard = scaler.transform(x_train)
    x_test_standard = scaler.transform(x_test)  # apply same transformation to test data
    # 3、单变量筛选
    print("step3: select topK cols...")
    selection = SelectKBest(mutual_info_regression, k=30)
    x_train_selection = selection.fit_transform(x_train_standard, y_train)
    x_test_selection = selection.transform(x_test_standard)
    selection_indices = selection.get_support(indices=True)  # 得到入模的列下标
    # 4、train model
    print("step4: train model...")
    estimator_num = 20
    clf = RandomForestRegressor(random_state=0, n_estimators=estimator_num, n_jobs=-1)
    clf.fit(x_train_selection, y_train)
    y_test_pred = clf.predict(x_test_selection)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("estimator_num={}, mse={}".format(estimator_num, mse))
    today = datetime.now().strftime("%Y%m%d")
    joblib.dump(clf, "./output/{}/rfRegressor_model.m".format(today))
    np.savetxt("./output/{}/mean_parameter.txt".format(today), scaler.mean_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/std_parameter.txt".format(today), scaler.var_[selection_indices], delimiter=",")
    np.savetxt("./output/{}/selection_indices.txt".format(today), selection_indices, delimiter=',', fmt="%d")
    np.savetxt("./output/{}/feature_importances.txt".format(today), clf.feature_importances_)

def initial():
    today = datetime.now().strftime("%Y%m%d")
    out_path = "./output/{}".format(today)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return today

def train_model(today):
    print("step1: delete some cols begin...")
    df = get_train_data()
    [rows, cols] = df.shape
    x = df.iloc[:, 0:cols - 1]
    y = df.iloc[:, cols - 1]
    id, enc_x = encode_onehot(x)
    x_train, x_test, y_train, y_test = train_test_split(enc_x, y, test_size=0.2, random_state=0)
    # 训练模型
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {'objective': 'reg:linear', 'silent': 1, 'eval_metric': ['error', 'rmse'], 'max_depth': 10, 'eta': .1}
    bst = xgb.train(param, dtrain, num_round, evallist)
    # fig = plt.figure()
    # plot_tree(bst, num_trees=3)
    # plt.show()
    # 预测
    y_train_pred = bst.predict(dtrain)
    y_test_pred = bst.predict(dtest)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    print("mse={}".format(mse))
    print(bst)
    joblib.dump(bst, "./output/{}/bstRegressor_model.m".format(today))


def predict_newdata():
    today = datetime.now().strftime("%Y%m%d")
    del_columns = get_delete_cols()
    input_data = pd.read_excel("./dataset/测试A.xlsx")
    del_df = input_data.drop(del_columns, axis=1)
    id, enc_df = encode_onehot(del_df)
    print(enc_df.columns)
    enc_df['TOOL_ID (#2)_E'] = 0
    enc_df['TOOL_C'] = 0
    enc_df['tool (#1)_T'] = 0
    sorted_cols = get_encode_schema()
    sorted_df = enc_df.reindex(columns=sorted_cols)
    sel_data = xgb.DMatrix(sorted_df)
    print(sorted_df.columns)
    # ＃加载模型
    bst = joblib.load("./output/{}/bstRegressor_model.m".format(today))
    predict_val = bst.predict(sel_data)
    result = list(zip(input_data.iloc[:, 0], predict_val))
    print(result)
    create_time = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt("./output/{}/训练A_{}.csv".format(today, create_time), result, fmt="%s", delimiter=",")

def test_tree_plot():
    from sklearn.datasets import load_iris
    from sklearn import tree
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    import graphviz
    dot_data = tree.export_graphviz(clf, out_file="./output/20171227/tree.pdf")
    graph = graphviz.Source(dot_data)
    graph.render("iris")

if __name__ == "__main__":
    today = initial()
    # getDataSummary("./dataset/训练.xlsx", "./output")
    # process_data()
    predict_newdata()
    # train_model(today)
    # calc_corr()
    # test_tree_plot()
