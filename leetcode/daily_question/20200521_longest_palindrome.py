#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 上午10:54
# @Author  : pengyuan.li
# @Site    : 
# @File    : 20200521_longest_palindrome.py
# @Software: PyCharm


class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        note：最长回文子串
        :param s:
        :return:
        """
        len_s = len(s)
        longest_str = ""
        longest_len = 0
        for i in range(len_s):
            for j in range(i, len_s):
                if self.is_huiwen(s, i, j) and (j - i + 1) > longest_len:
                    longest_len = j - i + 1
                    longest_str = s[i:j + 1]
        return longest_str

    def is_huiwen(self, s: str, start: int, end: int) -> bool:
        while start < end and s[start] == s[end]:
            start = start + 1
            end = end - 1
        return start >= end

    def longestPalindrome2(self, s: str) -> str:
        """
        note：最长回文子串，动态规划
        dp[i][j] = dp[i+1][j-1] and s[i]==s[j] , i < j i,j > 2
        dp[i][i] = True
        dp[i][i+1] = s[i]==s[i+1]
        :param s:
        :return:
        """
        len_s = len(s)
        dp = [[0 for _ in range(len_s)] for _ in range(len_s)]
        longest_str = ""
        longest_len = 0
        for i in range(len_s, -1, -1):
            for j in range(i, len_s):
                if i == j:
                    dp[i][j] = 1
                elif j - i == 1 and s[i] == s[j]:
                    dp[i][j] = 1
                else:
                    if dp[i + 1][j - 1] == 1 and s[i] == s[j]:
                        dp[i][j] = 1
                if dp[i][j] == 1 and j - i + 1 > longest_len:
                    longest_len = j - i + 1
                    longest_str = s[i:j + 1]
        print(dp)
        return longest_str


if __name__ == "__main__":
    ss = Solution()
    s = "babad"
    print(ss.longestPalindrome2(s))
    s = "b"
    print(ss.longestPalindrome2(s))
    s = "bbb"
    print(ss.longestPalindrome2(s))
    print(ss.is_huiwen(s, 0, len(s) - 1))
