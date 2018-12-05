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
    :type m: int
    :type n: int
    :rtype: int
    """
    path = [[1] * m] * n
    for i in range(1, n):
        for j in range(1, m):
            path[i][j] = path[i - 1][j] + path[i][j - 1]
    return path[n - 1][m - 1]


if __name__ == "__main__":
    m = 7
    n = 3
    print(unique_paths(m, n))
