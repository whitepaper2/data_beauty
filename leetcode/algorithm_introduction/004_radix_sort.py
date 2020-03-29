#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/29 下午4:20
# @Author  : pengyuan.li
# @Site    : 
# @File    : 004_radix_sort.py
# @Software: PyCharm
from typing import List


def radix_sort(nums):
    """
    note: 基数排序
    :param nums: 待排序数组
    :return:
    """
    ele_len = len(str(nums[0]))
    for i in range(ele_len - 1, -1, -1):
        # 分配十个list
        digit_list: List[List[int]] = [[], [], [], [], [], [], [], [], [], []]
        for j in range(len(nums)):
            ele = nums[j]
            for k in range(10):
                if int(str(ele)[i]) == k:
                    digit_list[k].append(ele)
        nums = [x for y in digit_list for x in y]
        print(nums)
        digit_list.clear()
    return nums


if __name__ == "__main__":
    nums = [3097, 3673, 2985, 1358, 6138, 9135, 4782, 1367, 3684, 1390]
    nums = radix_sort(nums)
    print(nums)
