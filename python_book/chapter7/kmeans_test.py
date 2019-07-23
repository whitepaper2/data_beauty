#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/8 下午3:25
# @Author  : pengyuan.li
# @Site    : 
# @File    : kmeans_test.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn import metrics  # 导入sklearn效果评估模块
from sklearn.cluster import KMeans  # KMeans模块
from sklearn.feature_extraction import DictVectorizer  # 字符串分类转整数分类库
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler库

# 读取数据
raw_data = pd.read_table('ad_performance.txt', delimiter='\t')
# 数据审查和校验
print('{:*^60}'.format('Data overview:'))
print(raw_data.head(2))  # 打印输出前2条数据
print('{:*^60}'.format('Data dtypes:'))
print(pd.DataFrame(raw_data.dtypes).T)  # 打印数据类型分布
print('{:*^60}'.format(' NA counts:'))
print(pd.DataFrame(raw_data.isnull().sum()).T)  # 查看缺失值情况
print('{:*^60}'.format('Data DESC:'))
print(raw_data.describe().round(2).T)  # 打印原始数据基本描述性信息
print('{:*^60}'.format('Correlation analysis:'))
print(raw_data.corr().round(2).T)  # 打印原始数据相关性信息

# 数据预处理
# 缺失值替换为均值
data_fillna = raw_data.fillna(raw_data['平均停留时间'].mean())  # 用均值替换缺失值

# 字符串分类转整数分类
# part1
conver_cols = ['素材类型', '广告类型', '合作方式', '广告尺寸', '广告卖点']
convert_matrix = data_fillna[conver_cols]  # 获得要转换的数组
lines = data_fillna.shape[0]  # 获得总记录数
dict_list = []  # 总空列表，用于存放字符串与对应索引组成的字典
unique_list = []  # 总唯一值列表，用于存储每个列的唯一值列表
# part2
for col_name in conver_cols:  # 循环读取每个列名
    cols_unqiue_value = data_fillna[col_name].unique().tolist()  # 获取列的唯一值列表
    unique_list.append(cols_unqiue_value)  # 将唯一值列表追加到总列表
# part3
for line_index in range(lines):  # 读取每行索引
    each_record = convert_matrix.iloc[line_index]  # 获得每行数据，是一个Series
    for each_index, each_data in enumerate(each_record):  # 读取Series每行对应的索引值
        list_value = unique_list[each_index]  # 读取该行索引对应到总唯一值列表列索引下的数据(其实是相当于原来的列做了转置成了行，目的是查找唯一值在列表中的位置)
        each_record[each_index] = list_value.index(each_data)  # 获得每个值对应到总唯一值列表中的索引
    each_dict = dict(zip(conver_cols, each_record))  # 将每个值和对应的索引组合字典
    dict_list.append(each_dict)  # 将字典追加到总列表
# part4
model_dvtransform = DictVectorizer(sparse=False, dtype=np.int64)  # 建立转换模型对象
data_dictvec = model_dvtransform.fit_transform(dict_list)  # 应用分类转换训练

# 数据标准化
sacle_matrix = data_fillna.ix[:, 1:8]  # 获得要转换的矩阵
minmax_scaler = MinMaxScaler()  # 建立MinMaxScaler模型对象
data_scaled = minmax_scaler.fit_transform(sacle_matrix)  # MinMaxScaler标准化处理

# 合并所有输入维度
X = np.hstack((data_scaled, data_dictvec))

# 通过平均轮廓系数检验得到最佳KMeans聚类模型
score_list = list()  # 用来存储每个K下模型的平局轮廓系数
score = []
for n_clusters in range(50):  # 遍历从2到10几个有限组
    model_kmeans = KMeans(n_clusters=4, init="random")  # 建立聚类模型对象
    cluster_labels_tmp = model_kmeans.fit_predict(X)  # 训练聚类模型
    silhouette_tmp = metrics.silhouette_score(X, cluster_labels_tmp)  # 得到每个K下的平均轮廓系数
    score_list.append([n_clusters, silhouette_tmp])  # 将每次K及其得分追加到列表
    score.append(silhouette_tmp)
print('{:*^60}'.format('K value and silhouette summary:'))
print(np.array(score_list))  # 打印输出所有K下的详细得分
print(score)


def get_init_random_performance(X):
    """
    在不同初始化向量下的k-means聚类效果
    :return: （随机数，平均轮廓系数）
    """
    pass


# if __name__ == "__main__":
#     pass
