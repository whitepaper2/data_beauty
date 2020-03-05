#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 下午3:17
# @Author  : pengyuan.li
# @Site    : 
# @File    : 098_isValidBST.py
# @Software: PyCharm
from typing import List


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def isValidBST(root: TreeNode) -> bool:
    """
    note：输入树的根节点，1、判断当前点是否满足；2、左右子树是否满足。
    这只能保证当前节点是BST，不能保证按照中序遍历二叉树是有序的。不符合题目要求。
    :param root:
    :return:
    """
    if root is None:
        return True
    if root.left.val >= root.val or root.right.val <= root.val:
        return False
    return isValidBST(root.left) and isValidBST(root.right)


def isValidBST2(root: TreeNode) -> bool:
    """
    note：输入树的根节点，1、判断当前点是否满足；2、左右子树是否满足。
    这只能保证当前节点是BST，不能保证按照中序遍历二叉树是有序的。不符合题目要求。
    :param root:
    :return:
    """
    if root is None:
        return True
    nums = []

    def inorder(root: TreeNode, nums: List[int]):
        if root is None:
            return
        inorder(root.left, nums)
        if root.val is not None:
            nums.append(root.val)
        inorder(root.right, nums)

    inorder(root, nums)
    for i in range(len(nums) - 1):
        if nums[i] >= nums[i + 1]:
            return False

    return True


def create_tree(nums):
    node_list = []
    for ele in nums:
        node = TreeNode(ele)
        node_list.append(node)
    for i in range(len(node_list) // 2):
        node_list[i].left = node_list[2 * i + 1]
        node_list[i].right = node_list[2 * i + 2]
    return node_list[0]


if __name__ == "__main__":
    nums = [2, 1, 3]
    root = create_tree(nums)
    print(isValidBST2(root))

    nums = [5, 1, 4, None, None, 3, 6]
    root = create_tree(nums)
    print(isValidBST2(root))
