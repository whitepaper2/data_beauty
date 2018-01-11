#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 17:13
# @Author  : Aries
# @Site    : 
# @File    : multi_process.py
# @Software: PyCharm

from multiprocessing import Process
import time
import multiprocessing
# 多线程并发
def task1(msg):
    print('task1: hello, %s' % msg)
    time.sleep(1)


def task2(msg):
    print('task2: hello, %s' % msg)
    time.sleep(1)


def task3(msg):
    print('task3: hello, %s' % msg)
    time.sleep(1)


if __name__ == '__main__':
    p1 = Process(target=task1, args=('one',))
    p2 = Process(target=task2, args=('two',))
    p3 = Process(target=task3, args=('three',))

    start = time.time()

    p1.start()
    p2.start()
    p3.start()

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child p.name: " + p.name + "\tp.id: " + str(p.pid))

    p1.join()
    p2.join()
    p3.join()

    end = time.time()
    print('3 processes take %s seconds' % (end - start))
# 多进程异步
from multiprocessing import Pool
import time


def task(msg):
    print('hello, %s' % msg)
    time.sleep(1)
    return 'msg: %s' % msg


if __name__ == '__main__':
    pool = Pool(processes=4)
    results = []
    # for x in range(10):
    #     ret = pool.apply_async(task, args=(x,))
    #     results.append(ret)
    msgs = [x for x in range(10)]
    results = pool.map(task, msgs)
    pool.close()
    pool.join()

    print('processes done.')
    for x in results:
        # print(x.get())
        print(x)
