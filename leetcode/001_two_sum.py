#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/9 下午10:25
# @Author  : pengyuan.li
# @Site    : 
# @File    : 001_two_sum.py
# @Software: PyCharm

def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    len_nums = len(nums)
    nums_dict = {}
    for i in range(len_nums):
        nums_dict[nums[i]] = i
    for i in range(len_nums):
        if nums[i] <= target or nums[i] <= abs(target):
            v1 = target - nums[i]
            if nums_dict.get(v1) is not None and i != nums_dict.get(v1):
                return [i, nums_dict.get(v1)]


if __name__ == '__main__':
    """
    Given nums = [2, 7, 11, 15], target = 9,
    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1]
    """
    nums = [2, 7, 11, 15]
    target = 9
    print(twoSum(nums, target))
