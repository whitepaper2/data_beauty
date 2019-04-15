#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 下午5:05
# @Author  : pengyuan.li
# @Site    : 
# @File    : 024_swap_pairs.py
# @Software: PyCharm

# Definition for singly-linked list.


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def swap_pairs(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy = ListNode(-1)
    pre = dummy
    pre.next = head
    while pre.next and pre.next.next:
        t = pre.next.next
        pre.next.next = t.next
        t.next = pre.next
        pre.next = t
        pre = t.next
    return dummy.next


if __name__ == "__main__":
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    node5 = ListNode(5)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    temp = node1
    # print(type(temp))
    while temp is not None:
        print(temp.val)
        temp = temp.next
    temp = swap_pairs(node1)
    while temp is not None:
        print(temp.val)
        temp = temp.next
