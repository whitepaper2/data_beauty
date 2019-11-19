#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 下午3:05
# @Author  : pengyuan.li
# @Site    : 
# @File    : 054_spiral_order.py
# @Software: PyCharm

# 顺时针访问矩阵
from typing import List


def spiral_order(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    m = len(matrix)
    n = len(matrix[0])
    top = 0
    bottom = m - 1
    left = 0
    right = n - 1
    result = []
    side = 0
    while top <= bottom and left <= right:
        if side == 0:
            i = top
            for j in range(left, right+1):
                result.append(matrix[i][j])
            top = top + 1
        if side == 1:
            j = right
            for i in range(top, bottom+1):
                result.append(matrix[i][j])
            right = right - 1
        if side == 2:
            i = bottom
            for j in range(right, left-1, -1):
                result.append(matrix[i][j])
            bottom = bottom - 1
        if side == 3:
            j = left
            for i in range(bottom, top-1, -1):
                result.append(matrix[i][j])
            left = left + 1
        side = (side + 1) % 4
    return result


if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(spiral_order(matrix))
