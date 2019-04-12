#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 上午10:24
# @Author  : pengyuan.li
# @Site    : 
# @File    : feature_importance.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# 1\ common method to calculate feature importances
def woe_iv(data, feature, label):
    # :param data: DataFrame, data files contain feature label
    # :param feature: String, a predict variable
    # :param label: String, the object variable
    # :return: information value of given feature
    data = data[[feature, label]]
    cato_num = data.groupby(feature).count().reset_index()
    default_num = data.groupby(feature).sum().reset_index()
    all_number = data.shape[0]
    default_number = data[label].sum()
    normal_number = all_number - default_number
    iv = 0
    for i in np.arange(cato_num.shape[0]):
        p_default = default_num[label][i] / default_number
        p_normal = (cato_num[label][i] - default_num[label][i]) / normal_number
        if p_default == 0 or p_normal == 0:
            print("woe_{}_{} is not available".format(feature, cato_num[feature][i]))
        else:
            locals()["woe_{}".format(cato_num[feature][i])] = np.log(p_normal / p_default)
            print("woe_{}: {}".format(cato_num[feature][i], locals()["woe_{}".format(cato_num[feature][i])]))
            iv = iv + (p_normal - p_default) * locals()["woe_{}".format(cato_num[feature][i])]
    print("iv={} of {}".format(feature, iv))
    return iv


def plot_bar():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    data = []
    # data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,0,0,0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(100):
        data.append(random.randint(0, 1))
    step = float(1 / len(data))

    l = list(range(4))
    x = 0.0
    cnt = 0
    print(data)
    # step = 0.1
    for i in data:
        if i == 0:
            ax1.add_patch(patches.Rectangle((x, 0), step, 0.6, facecolor="white"))
        if i == 1:
            ax1.add_patch(patches.Rectangle((x, 0), step, 0.6, facecolor="black"))
        cnt = cnt + 1
        x = x + step
        print(x)
        # x += 0.012
    fig1.savefig('./barcode.png', dpi=90, bbox_inches='tight')
    plt.close()


def func_args(input, *args, **kwargs):
    print(input)
    print(args)
    print(args[0])
    for arg in args:
        print("name={}".format(arg))
    print(kwargs)


def main(input, *args, **kwargs):
    print(args)
    context = {}
    context["hdfs_url"] = "10.57"
    context["login_account_id"] = "10.57.http"
    args = ("10.57", "http:pengyuan")
    kwargs.update(context)
    func_args(input, *args, **kwargs)


if __name__ == "__main__":
    hdfs_dict = {}
    hdfs_dict["hdfs_url"] = "10.57"
    hdfs_dict["login_account_id"] = "10.57.http"

    input = 1

    # args = [context.get("hdfs_url"), context.get("login_account_id")]
    main(input, hdfs_dict, 2, 3)
    # plot_bar()
    # 生成data，包括x1,x2,x3三个自变量
    # x1 = np.random.randint(-3, 3, 1000)
    # x2 = 1.5 * np.random.randint(-3, 3, 1000)
    # x3 = 0.5 * np.random.randint(-3, 3, 1000)
    # y = (1 + x1 + x2 + x3 + np.random.randn()) > 0
    # X = np.column_stack([x1, x2, x3])
    # data = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    # data["y"] = y
    # woe_iv(data, "x1", "y")

# 2\ linear model feature importances


x1 = np.random.randn(100)
x2 = 4 * np.random.randn(100)
x3 = 0.5 * np.random.randn(100)
y = (3 + x1 + x2 + x3 + 0.2 * np.random.randn()) > 0
X = np.column_stack([x1, x2, x3])
m = LogisticRegression()
m.fit(X, y)  # 逻辑回归预计的参数都接近1:print(m.coef_)# 以下输出说明了X2有更强的预测可以力，符合预期print(np.std(X, 0)m.coef_)
print(m.coef_)
print(np.std(X, 0) * m.coef_)
