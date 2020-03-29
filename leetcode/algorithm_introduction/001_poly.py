#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 下午10:38
# @Author  : pengyuan.li
# @Site    : 
# @File    : 001_poly.py
# @Software: PyCharm

def poly(A, x, n):
    """
    note:多项式计算方法，honer法则
    :param A:
    :param x:
    :param n:
    :return:
    """
    res = A[0]
    for i in range(1, n):
        res = res * x + A[i]
    return res


if __name__ == "__main__":
    A = [2, 3, 4, 1]
    n = 4
    x = 2

    print(poly(A, x, n))
