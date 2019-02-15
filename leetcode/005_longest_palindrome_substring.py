#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/14 下午2:53
# @Author  : pengyuan.li
# @Site    : 
# @File    : 005_longest_palindrome_substring.py
# @Software: PyCharm


def longest_palindrome(s):
    """
    note: Time Limit Exceeded，循环遍历s[i:j]是否为回文字符串
    :type s: str
    :rtype: str
    """

    def is_palindrome(ss, i, j):
        out = True
        x_str = str(ss[i:j])
        left = 0
        right = len(x_str) - 1
        while left < right:
            if x_str[left] == x_str[right]:
                left = left + 1
                right = right - 1
            else:
                out = False
                break
        return out

    res_len = 0
    res_i = 0
    res_j = 0
    for i in range(0, len(s)):
        for j in range(i + 1, len(s) + 1):
            if is_palindrome(s, i, j) and j - i > res_len:
                res_len = j - i
                res_i = i
                res_j = j
    return s[res_i:res_j]


def longest_palindrome2(s):
    """
    note: 动态规划求回文字符串
    :type s: str
    :rtype: str
    """
    res_left = 0
    res_right = 0
    res_len = 0
    dp = []
    for j in range(len(s)):
        dp.append([0] * len(s))
        dp[j][j] = 1
        for i in range(0, j):
            # if j - i > 1:
            #     dp[i][j] = 1 if s[i] == s[j] and dp[i + 1][j - 1] else 0
            # else:
            #     dp[i][j] = 1 if s[i] == s[j] else 0
            dp[i][j] = 1 if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]) else 0
            if dp[i][j] and res_len < j - i + 1:
                res_len = j - i + 1
                res_left = i
                res_right = j

    return s[res_left:res_right + 1]


if __name__ == "__main__":
    s = "babad"
    print(longest_palindrome2(s))

    s = "cbbd"
    print(longest_palindrome2(s))

    s = ""
    print(longest_palindrome2(s))
