#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 下午2:06
# @Author  : pengyuan.li
# @Site    : 
# @File    : feature_engineer.py
# @Software: PyCharm
# 文本分析
import preprocess as pre
import jieba
import jieba.analyse

text_path = "./dataset/meinian_round1_data_part1_20180408.txt"
split_words_path = "./output/{}/split_words.txt".format(pre.today)
words_freq_path = "./output/{}/words_freq.txt".format(pre.today)


def get_split_words():
    wf = open(split_words_path, 'w+')
    for line in open(text_path):
        item = line.strip('\n\r').split('$')
        # print item[1]
        tags = jieba.analyse.extract_tags(item[2])
        tagsw = ",".join(tags)
        wf.write(tagsw)

    wf.close()


def get_words_freq():
    word_lst = []
    word_dict = {}
    with open(split_words_path) as wf, open(words_freq_path, 'w') as wf2:

        for word in wf:
            word_lst.append(word.split(','))
            for item in word_lst:
                for item2 in item:
                    if item2 not in word_dict:
                        word_dict[item2] = 1
                    else:
                        word_dict[item2] += 1

        for key in word_dict:
            # print(key, word_dict[key])
            wf2.write(key + ' ' + str(word_dict[key]) + '\n')


if __name__ == "__main__":
    # get_split_words()
    get_words_freq()