#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 下午6:58
# @Author  : pengyuan.li
# @Site    : 
# @File    : common.py
# @Software: PyCharm


import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        start = time.time()
        data = func(*args, **kwds)
        end = time.time()
        print('function {} used time :{}ms'.format(func.__name__, (end - start) * 1000))
        # print('function {} used time :{}ms'.format(func.__name__, int(round((end - start) * 1000))))
        return data

    return wrapper
