#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 下午5:55
# @Author  : pengyuan.li
# @Site    : 
# @File    : 039_combination_sum.py
# @Software: PyCharm


def combination_sum(candidates, target):
    """
    note: 递增排序，深度优先遍历，从当前数据点开始计算，target = target-a[i]
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    from copy import deepcopy
    out = []
    sort_candidates = sorted(candidates)
    cur_out = []

    def sub_combination(sort_candidates, target, start, cur_out, out):
        if target < 0:
            return
        elif target == 0:
            out.append(deepcopy(cur_out))
            return
        else:
            for i in range(start, len(sort_candidates)):
                cur_out.extend([sort_candidates[i]])
                sub_combination(sort_candidates, target - sort_candidates[i], i, cur_out, out)
                cur_out.pop()

    sub_combination(sort_candidates, target, 0, cur_out, out)
    return out


if __name__ == "__main__":
    candidates = [2, 3, 6, 7]
    target = 7
    print(combination_sum(candidates, target))

    candidates = [2, 3, 5]
    target = 8
    print(combination_sum(candidates, target))
