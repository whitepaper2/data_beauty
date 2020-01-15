#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 下午2:21
# @Author  : pengyuan.li
# @Site    : 
# @File    : 062_unique_paths.py
# @Software: PyCharm


def unique_paths(m, n):
    """
    note: 大问题拆分为子问题，矩阵表示与常规正好相反
    动态规划，机器人只能向右或向下移动，dp[i][j]表示从左上角到坐标(i, j)的路径长度，状态转移公式：dp[i][j]=dp[i-1][j]+dp[i][j-1]
    :type m: int，列数
    :type n: int，行数
    :rtype: int，不同路径的个数
    """
    path = [[1] * m] * n
    for i in range(1, n):
        for j in range(1, m):
            path[i][j] = path[i - 1][j] + path[i][j - 1]
    return path[n - 1][m - 1]


def unique_paths2(m, n):
    """
    note: 大问题拆分为子问题，节省空间，保存一维数据（不太明白？？？），每行保存一个值，更新方式：d[i] = d[i] + d[i-1]
    动态规划，机器人只能向右或向下移动，dp[i][j]表示从左上角到坐标(i, j)的路径长度，状态转移公式：dp[i][j]=dp[i-1][j]+dp[i][j-1]
    :type m: int，列数
    :type n: int，行数
    :rtype: int，不同路径的个数
    """
    path = [1] * n
    for j in range(1, m):
        for i in range(1, n):
            path[i] = path[i - 1] + path[i]
    return path[n - 1]


if __name__ == "__main__":
    m = 7
    n = 3
    print(unique_paths(m, n))
    print(unique_paths2(m, n))
