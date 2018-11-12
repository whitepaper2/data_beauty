#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 上午9:15
# @Author  : pengyuan.li
# @Site    : 
# @File    : 009_is_palindrome.py
# @Software: PyCharm


def is_palindrome(x):
    """
    :type x: int
    :rtype: bool
    """
    x_out = []
    out = True
    if x < 0:
        out = False
    else:
        while x >= 1:
            x_res = x % 10
            x_int = int(x / 10)
            x = x_int
            x_out.append(x_res)
        len_xout = len(x_out)
        for i in range(int(len_xout / 2)):
            if x_out[i] != x_out[len_xout - i - 1]:
                out = False
    return out


if __name__ == "__main__":
    """
    Given x = 121, return True
    Given x = -121, return False
    Given x = 120, return False
    """
    print(is_palindrome(0))
    print(is_palindrome(121))
    print(is_palindrome(-121))
    print(is_palindrome(120))
