#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/9 下午4:07
# @Author  : pengyuan.li
# @Site    : 
# @File    : 215_find_kth_largest.py
# @Software: PyCharm
from typing import List


def findKthLargest(nums: List[int], k: int) -> int:
    # 建立k个元素的最小堆，如果堆顶元素<原数组值，则替换，然后调整堆结构。最后返回堆顶元素
    len_nums = len(nums)
    k_nums = build_minheap(nums[:k], k)
    for i in range(k, len_nums):
        if k_nums[0] < nums[i]:
            k_nums[0] = nums[i]
            adjust_minheap(k_nums, 0, k)
    return k_nums[0]


def adjust_maxheap(nums: List[int], i: int, n: int):
    """
    调整最大堆的结构
    :param nums: 原始数组
    :param i: 第i个节点
    :param n: 数组大小
    :return:
    """
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and nums[left] > nums[i]:
        largest = left
    else:
        largest = i
    if right < n and nums[right] > nums[largest]:
        largest = right
    if largest != i:
        tmp = nums[largest]
        nums[largest] = nums[i]
        nums[i] = tmp
        adjust_maxheap(nums, largest, n)


def build_maxheap(nums: List[int], n: int):
    """
    自下而上建立最大堆
    :param nums: 原始数组
    :param n: 原始数组大小
    :return:
    """
    for i in range(n // 2, -1, -1):
        adjust_maxheap(nums, i, n)


def heap_sorting(nums: List[int]):
    """
    按递增排列数组，因为最大堆堆顶总是最大值，应该放在数组尾部，每次交换堆顶元素和之前的元素
    :param nums:
    :return:
    """
    n = len(nums)
    build_maxheap(nums, n)
    for i in range(n - 1, 0, -1):
        tmp = nums[0]
        nums[0] = nums[i]
        nums[i] = tmp
        adjust_maxheap(nums, 0, i)
    return nums


def adjust_minheap(nums: List[int], i: int, n: int):
    """
    调整最小堆
    :param nums:原始数组
    :param i: 当前第i个数据
    :param n: 数组大小
    :return:
    """
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and nums[left] < nums[i]:
        min_ind = left
    else:
        min_ind = i
    if right < n and nums[right] < nums[min_ind]:
        min_ind = right
    if min_ind != i:
        tmp = nums[i]
        nums[i] = nums[min_ind]
        nums[min_ind] = tmp
        adjust_minheap(nums, min_ind, n)


def build_minheap(nums: List[int], n: int):
    """
    建立最小堆
    :param nums:原始数组
    :param n: 数组大小
    :return:
    """
    for i in range(n // 2, -1, -1):
        adjust_minheap(nums, i, n)
    return nums


if __name__ == "__main__":
    nums = [3, 2, 1, 5, 6, 4]
    build_maxheap(nums, len(nums))
    print(nums)
    heap_sorting(nums)
    print(nums)
    k = 2
    print(findKthLargest(nums, k))

    nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
    k = 4
    print(findKthLargest(nums, k))

    nums = [2, 1]
    k = 2
    print(findKthLargest(nums, k))
