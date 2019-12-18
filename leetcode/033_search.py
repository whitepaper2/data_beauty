#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 下午4:23
# @Author  : pengyuan.li
# @Site    : 
# @File    : 033_search.py
# @Software: PyCharm
# 递增序列可以使用二分法求解，按照枢纽点遍历所有的序列，找到规律，在枢纽点前面或后面都是有序排列
from typing import List


def search(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[right] >= target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


if __name__ == "__main__":
    nums = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    print(search(nums, target))
    target = 3
    print(search(nums, target))
    nums = [3, 5, 1]
    target = 3
    print(search(nums, target))
    nums = [1, 3]
    target = 3
    print(search(nums, target))
