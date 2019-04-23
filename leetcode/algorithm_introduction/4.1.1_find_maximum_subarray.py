#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 下午3:24
# @Author  : pengyuan.li
# @Site    : 
# @File    : 4.1.1_find_maximum_subarray.py
# @Software: PyCharm


def brute_find_maximum_subarray(stocks):
    """
    note: 暴力搜索，时间复杂度 n^2，先买入，再卖出
    :param stocks: 一段时间内的股票交易价格
    :return: 买入时间、卖出时间、赚的差价
    """
    max_sum = -float("inf")
    min_in = 0
    max_out = len(stocks) - 1
    for i in range(len(stocks)):
        for j in range(i, len(stocks)):
            if stocks[i] <= stocks[j] and stocks[j] - stocks[i] > max_sum:
                max_sum = stocks[j] - stocks[i]
                min_in = i
                max_out = j
    return min_in, max_out, max_sum


def divide_find_maximum_subarray(stocks):
    """
    note: 分治搜索，时间复杂度 n，先买入，再卖出。
    问题变换，原数据转为相对于交易前一天的差值，求最大子数组和即得连续和。
    :param stocks: 一段时间内的股票交易价格
    :return: 买入时间、卖出时间、赚的差价
    """
    diff_stocks = [stocks[i] - stocks[i - 1] for i in range(1, len(stocks))]
    left = 0
    right = len(diff_stocks) - 1

    def find_cross_maximum(left, mid, right, diff_stocks):
        """
        note: 求包含mid的连续子数组
        :param left: 左边起始点
        :param mid: 中间点
        :param right: 右边终点
        :param diff_stocks: 与上一日比的交易价格差
        :return: 买入时间、卖出时间、赚的差价
        """
        max_left = left
        max_right = right
        right_sum = -float("inf")
        cur_sum = 0
        for i in range(mid + 1, right + 1):
            cur_sum = cur_sum + diff_stocks[i]
            if cur_sum > right_sum:
                right_sum = cur_sum
                max_right = i
        left_sum = -float("inf")
        cur_sum = 0
        for i in range(mid, left, -1):
            cur_sum = cur_sum + diff_stocks[i]
            if cur_sum > left_sum:
                left_sum = cur_sum
                max_left = i

        return max_left, max_right, left_sum + right_sum

    def find_maximum_subarray(left, right, diff_stocks):
        """
        note: 最大连续子数组和分为三种情况：1）落在左边；2）落在右边；3）包含mid
        :param left: 左边起始点
        :param right: 右边终点
        :param diff_stocks: 相对于前一天股票价格的差值
        :return: 买入时间、卖出时间、赚的差价
        """
        if right == left:
            return left, right, diff_stocks[left]
        else:
            mid = (left + right) // 2
            left_in, left_out, left_value = find_maximum_subarray(left, mid, diff_stocks)
            right_in, right_out, right_value = find_maximum_subarray(mid + 1, right, diff_stocks)
            cross_in, cross_out, cross_value = find_cross_maximum(left, mid, right, diff_stocks)
            if left_value >= right_value and left_value >= cross_value:
                return left_in, left_out, left_value
            elif right_value >= left_value and right_value >= cross_value:
                return right_in, right_out, right_value
            else:
                return cross_in, cross_out, cross_value

    min_in, max_out, max_sum = find_maximum_subarray(left, right, diff_stocks)

    return min_in, max_out + 1, max_sum


if __name__ == "__main__":
    stocks = [100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]
    sell_in, sell_out, price = brute_find_maximum_subarray(stocks)
    print(sell_in, sell_out, price)
    sell_in, sell_out, price = divide_find_maximum_subarray(stocks)
    print(sell_in, sell_out, price)
