#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 下午3:27
# @Author  : pengyuan.li
# @Site    : 
# @File    : vikor.py
# @Software: PyCharm

# 多属性评价法，选出最优企业

import numpy as np
import pandas as pd

input_df = pd.read_csv("./dataset/vikor_test.csv")
# v4越小越优，其他变量越大越优
add_column = ["v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11"]
subtract_column = ["v4"]
select_column = add_column + subtract_column
(m, n) = input_df.shape
# step1.利用变异系数法计算指标权重w
a = input_df[select_column].mean()
s = input_df[select_column].std()  # 样本标准差N-1
delta = s / a
w = delta / delta.sum()
# step2.计算决策矩阵r，分为收益型和成本型
r = pd.DataFrame(np.zeros([m, len(select_column)]), columns=select_column)
r[add_column] = (input_df[add_column] - input_df[add_column].min()) / (
        input_df[add_column].max() - input_df[add_column].min())
r[subtract_column] = (input_df[subtract_column].max() - input_df[subtract_column]) / (
        input_df[subtract_column].max() - input_df[subtract_column].min())
# step3.确定理想点和负理想点
r_star = r.max()
r_star2 = r.min()
# step4.计算群体效用值和个别遗憾值
ss = np.zeros([m])
rr = np.zeros([m])
for i in range(m):
    ss[i] = (w * (r_star - r.iloc[i][select_column]) / (r_star - r_star2)).sum()
    rr[i] = (w * (r_star - r.iloc[i][select_column]) / (r_star - r_star2)).max()
v = 0.5
ss_star = ss.min()
# step5.计算综合排序指标，指标越小，信用等级越高
q = np.zeros([m])
for i in range(m):
    q[i] = v * (ss[i] - ss.min()) / (ss.max() - ss.min()) + (1 - v) * (rr[i] - rr.min()) / (rr.max() - rr.min())
# q.sort()
print(q)
