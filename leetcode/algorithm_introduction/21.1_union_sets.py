#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 下午8:45
# @Author  : pengyuan.li
# @Site    : 
# @File    : 21.1_union_sets.py
# @Software: PyCharm


class UnionFindSets:
    def __init__(self, A):
        """
        维护父节点(字典)、高度(字典)、并查集个数
        :param A: 输入list
        """
        self.parent = {}
        self.rank = {}
        self.setsNum = len(A)
        for a in A:
            self.parent[a] = a
            self.rank[a] = 1

    def find(self, x):
        father = self.parent[x]
        # while father != x:
        # father = self.parent[father]
        if father != x:
            father = self.find(father)
        self.parent[x] = father
        return father

    def union(self, x, y):
        if not x or not y:
            return
        xHead = self.find(x)
        yHead = self.find(y)
        if xHead == yHead:
            return

        xRank = self.rank[xHead]
        yRank = self.rank[yHead]
        if xRank >= yRank:
            self.parent[yHead] = xHead
            if xRank == yRank:
                self.rank[xRank] += 1
        else:
            self.parent[xHead] = yHead
        self.setsNum -= 1


if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]
    ufs = UnionFindSets(nums)
    ufs.union(1, 3)
    print(ufs.parent, ufs.rank, ufs.setsNum)

    ufs.union(1, 3)
    print(ufs.parent, ufs.rank, ufs.setsNum)
