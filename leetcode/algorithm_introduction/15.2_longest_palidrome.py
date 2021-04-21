#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/26 下午8:53
# @Author  : pengyuan.li
# @Site    : 
# @File    : 16.1_greedy_activity.py
# @Software: PyCharm


def lcs(X):
    """
    最长回文子串，c[i,j]=c[i+1,j-1]+1, X[i]==Y[j]
    c[i,j]=max(c[i+1,j],c[i,j-1]，X[i]!=Y[j]
    :param X:
    :return:
    """
    m = len(X)
    c = [[0] * (m + 1) for _ in range(m + 1)]
    for i in range(m, 0, -1):
        for j in range(i, m + 1):
            if i == j:
                c[i][j] = 1
            elif j - i == 1:
                c[i][j] = 2 if X[i - 1] == X[j - 1] else 1
            elif j - i == 2:
                c[i][j] = 3 if X[i - 1] == X[j - 1] else max(c[i][j - 1], c[i + 1][j])
            else:
                if X[i + 1] == X[j - 1]:
                    c[i][j] = c[i + 1][j - 1] + 1
                else:
                    c[i][j] = max(c[i][j - 1], c[i + 1][j])
    print(c)
    return c[1][m]


def printLCS(p, X, m, n):
    """
    打印出共同的最长字符串
    :param p: 记录的路径
    :param X:
    :param m: x.length
    :param n: y.length
    :return:
    """
    if m == 0 or n == 0:
        return
    if p[m][n] == "tri":
        print(X[m - 1])
        printLCS(p, X, m - 1, n - 1)
    elif p[m][n] == "left":
        printLCS(p, X, m, n - 1)
    else:
        printLCS(p, X, m - 1, n)


def dpLCS(A):
    """
    最长单调子序列，dp[i]=max(dp[j])+1,j<i,A[j]<A[i]
    :param A:
    :return:
    """
    n = len(A)
    dp = [1] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        # tmp = dp[i]
        for j in range(i - 1, 0, -1):
            if A[j - 1] < A[i - 1] and dp[i] < dp[j] + 1:
                # if tmp < dp[j] + 1:
                #     tmp = dp[j] + 1
                dp[i] = dp[j] + 1
    print(dp)
    return max(dp)


if __name__ == "__main__":
    s = "character"
    print(lcs(s))
