#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 下午8:18
# @Author  : pengyuan.li
# @Site    : 
# @File    : 064_min_path_sum.py
# @Software: PyCharm
from typing import List
import copy


def minPathSum(grid: List[List[int]]) -> int:
    """
    pathsum[i][j] = path[i][j] + min(pathsum[i-1][j], pathsum[i][j-1])
    :param grid: 路径数据
    :return: int，最小路径和
    """
    if grid is None:
        return 0
    m = len(grid)
    n = len(grid[0])
    if m == 0 or n == 0:
        return 0
    psum = copy.deepcopy(grid)

    for i in range(1, n):
        psum[0][i] = psum[0][i - 1] + psum[0][i]
    for i in range(1, m):
        psum[i][0] = psum[i - 1][0] + psum[i][0]
    for i in range(1, m):
        for j in range(1, n):
            psum[i][j] = grid[i][j] + min(psum[i - 1][j], psum[i][j - 1])
    return psum[m - 1][n - 1]


def minPathSum2(grid: List[List[int]]) -> int:
    """
    pathsum[i][j] = path[i][j] + min(pathsum[i-1][j], pathsum[i][j-1])
    :param grid: 路径数据
    :return: int，最小路径和
    """
    if grid is None:
        return 0
    m = len(grid)
    n = len(grid[0])
    if m == 0 or n == 0:
        return 0
    psum = [0] * m
    psum[0] = grid[0][0]
    for i in range(1, m):
        psum[i] = psum[i - 1] + grid[i][0]
    for i in range(1, n):
        for j in range(0, m):
            if j == 0:
                psum[j] = psum[j] + grid[0][i]
            else:
                psum[j] = min(psum[j - 1], psum[j]) + grid[j][i]
    return psum[m - 1]


if __name__ == "__main__":
    grid = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]
    print(minPathSum(grid))
    print(minPathSum2(grid))
