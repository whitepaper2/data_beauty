#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/29 下午9:45
# @Author  : pengyuan.li
# @Site    : 
# @File    : 005_is_heap.py
# @Software: PyCharm


def is_heap(nums):
    """
    note: 判断n个元素的数组是否为堆结构
    :param nums:
    :return: True/False
    """
    flag = True
    n = len(nums)
    for i in range(n // 2, -1, -1):
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        least = i
        if left < n:
            if nums[i] < nums[left]:
                largest = left
            if nums[i] > nums[left]:
                least = left
        if right < n:
            if nums[i] < nums[right]:
                largest = right
            if nums[i] > nums[right]:
                least = right
        # 只要有一个成立，则为堆
        if largest != i and least != i:
            flag = False
            break
    return flag


if __name__ == "__main__":
    nums = [9, 4, 8, 3, 2, 5, 7]
    print(is_heap(nums))
    nums = [9, 4, 7, 2, 1, 6, 5, 3]
    print(is_heap(nums))
    nums = [9]
    print(is_heap(nums))
    nums = [8, 6, 4, 3, 2]
    print(is_heap(nums))
    nums = [9, 7, 5, 6, 3]
    print(is_heap(nums))
