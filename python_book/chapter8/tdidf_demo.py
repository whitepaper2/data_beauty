#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 上午9:56
# @Author  : pengyuan.li
# @Site    : 
# @File    : tdidf_demo.py
# @Software: PyCharm


#
# corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
#           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
#           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
#           [(0, 1.0), (4, 2.0), (7, 1.0)],
#           [(3, 1.0), (5, 1.0), (6, 1.0)],
#           [(9, 1.0)],
#           [(9, 1.0), (10, 1.0)],
#           [(9, 1.0), (10, 1.0), (11, 1.0)],
#           [(8, 1.0), (10, 1.0), (11, 1.0)]]
#
# tfidf = models.TfidfModel(corpus)
# vec = [(0, 3), (4, 2)]
# out = tfidf[vec]
# print(tfidf[vec])

# os.chdir("D:\\Python_config")

# cf = configparser.ConfigParser()
#
# # cf.read("test.ini")
# cf.read("db.conf")
#
# # return all section
# secs = cf.sections()
# print('sections:', secs, type(secs))
# opts = cf.options("db")
# print('options:', opts, type(opts))
# kvs = cf.items("db")
# print('db:', kvs)
#
# # read by type
# db_host = cf.get("db", "db_host")
# db_port = cf.getint("db", "db_port")
# db_user = cf.get("db", "db_user")
# db_pass = cf.get("db", "db_pass")
#
# # read int
# threads = cf.getint("concurrent", "thread")
# processors = cf.getint("concurrent", "processor")
# print("db_host:", db_host)
# print("db_port:", db_port)
# print("db_user:", db_user)
# print("db_pass:", db_pass)
# print("thread:", threads)
# print("processor:", processors)


def reverse(x):
    """
    :type x: int
    :rtype: int
    """
    print(pow(2, 31))
    x_out = []
    is_negative = False
    if x == 0:
        out = 0
    else:
        if x < 0:
            x = abs(x)
            is_negative = True
        num = 0
        while x >= 1:
            num = num + 1
            x_res = x % 10
            x_int = int(x / 10)
            x = x_int
            x_out.append(x_res)
        out = x_out[0]
        if num > 1:
            for i in range(num - 1):
                out = out * 10 + x_out[i + 1]
                if out > pow(2, 31) - 1:
                    out = 0
                    break
        if is_negative:
            out = -1 * out
    return out


def fib(max):
    a, b = 1, 1
    while a < max:
        yield a
        a, b = b, a + b


if __name__ == '__main__':
    # res = twoSum([-1, -2, -3, -4, -5, -6], -8)
    # res = twoSum([-1, -2, 1, 4, -5, -6], 0)
    res = reverse(123)
    print(res)
    res = reverse(-123)
    print(res)
    res = reverse(120)
    print(res)
import networkx as nx

G = nx.Graph()
nodelist = ['Enoch', 'Evan', 'Gregary', 'Hugo',
            'Jeff', 'Hale', 'Keith', 'Leif',
            'Mick', 'Noah', 'Lionel', 'Rex',
            'Parker', 'Stan']  # the list of nodes
for i in nodelist:  # add nodes to graph G
    G.add_node(i)
edgelist = [['Enoch', 'Evan'], ['Enoch', 'Gregary'], ['Evan', 'Hugo'],
            ['Evan', 'Jeff'], ['Gregary', 'Hale'], ['Gregary', 'Keith'],
            ['Jeff', 'Hale'], ['Jeff', 'Keith'], ['Keith', 'Leif'],
            ['Keith', 'Lionel'], ['Leif', 'Mick'], ['Mick', 'Noah'],
            ['Noah', 'Lionel'], ['Rex', 'Parker'], ['Parker', 'Stan'], ['Rex', 'Stan']]
for i in edgelist:
    G.add_edge(i[0], i[1])

nx.pagerank(G)
score = nx.betweenness_centrality(G)
print(score)
# from networkx.algorithms.community import label_propagation
# res = label_propagation.asyn_lpa_communities(G)
# nx.betweenness_centrality()
# while True:
#     try:
#         each = next(res)
#     except StopIteration:
#         break
#     print(each)
