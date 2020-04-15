#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 下午2:01
# @Author  : pengyuan.li
# @Site    : 
# @File    : 003_train_seq2seq.py
# @Software: PyCharm

from random import randint

import pandas as pd
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from keras.utils import to_categorical
from numpy import argmax
from numpy import array
from numpy import array_equal

# sampled_token_index = argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]

feature_col = ["carflowcnt", "last1dcnt", "last2dcnt", "last3dcnt", "last4dcnt", "last5dcnt", "last6dcnt",
               "last7dcnt",
               "last8dcnt", "last9dcnt", "last10dcnt", "last11dcnt", "last12dcnt", "last13dcnt", "last14dcnt"]
label_col = ["next1dcnt", "next2dcnt", "next3dcnt", "next4dcnt", "next5dcnt"]


def get_train_data():
    pdf = pd.read_csv("./datasets/basic_route_hour_carcnt_label.csv")
    pdf = pdf[pdf["hour_time"] == 10]
    pdf = pdf.fillna(0)
    return pdf[pdf["day_time"] <= 20200225]


def get_test_data():
    pdf = pd.read_csv("./datasets/basic_route_hour_carcnt_label.csv")
    pdf = pdf[pdf["hour_time"] == 10]
    pdf = pdf.fillna(0)
    return pdf[pdf["day_time"] > 20200225]


def get_flowcnt_dict():
    pdf = pd.read_csv("./datasets/basic_route_hour_carcnt_label.csv")
    pdf = pdf[pdf["hour_time"] == 10]
    pdf = pdf.fillna(0)

    unique_flow = set()
    for col in feature_col + label_col:
        carflow = pdf[col].unique()
        for i in carflow:
            unique_flow.add(i)
    flow_list = sorted(list(unique_flow))
    flow_dict = dict([(char, i) for i, char in enumerate(flow_list)])
    return flow_dict


# 构造LSTM模型输入需要的训练数据
def get_dataset(pdf, flow_dict):
    X1, X2, y = list(), list(), list()
    one_hot_len = len(flow_dict)
    for i, r in pdf.iterrows():
        source = [flow_dict.get(x) for x in r[feature_col]]
        target = [flow_dict.get(x) for x in r[label_col]]
        target_in = [0] + target[:-1]
        src_encoded = to_categorical(source, num_classes=one_hot_len)
        tar_encoded = to_categorical(target, num_classes=one_hot_len)
        tar2_encoded = to_categorical(target_in, num_classes=one_hot_len)

        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    return array(X1), array(X2), array(y)


# 构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
def define_models(n_input, n_output, n_units):
    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量
    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # 返回需要的三个模型
    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, source, n_steps, cardinality, reverse_flow_dict):
    # 输入序列编码得到编码状态向量
    state = infenc.predict(source)
    # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # 输出序列列表
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        print(yhat)
        print(len(yhat))
        # 截取输出序列，取后三个
        # output.append(yhat[0, 0, :])
        encoded_seq_ind = argmax(yhat[0, -1, :])
        output.append(reverse_flow_dict.get(encoded_seq_ind))
        # 更新状态
        state = [h, c]
        # 更新目标序列(用于下一个词预测的输入)
        target_seq = yhat
    return array(output)


# one_hot解码
def one_hot_decode(encoded_seq, reverse_flow_dict):
    return


def train_model():
    # 参数设置
    flow_dict = get_flowcnt_dict()
    n_features = len(flow_dict)
    reverse_flow_dict = dict((i, char) for char, i in flow_dict.items())
    # 定义模型
    train, infenc, infdec = define_models(n_features, n_features, 128)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # 生成训练数据

    X1, X2, y = get_dataset(get_train_data(), flow_dict)
    print(X1.shape, X2.shape, y.shape)
    # 训练模型
    train.fit([X1, X2], y, epochs=1)
    # 评估模型效果
    total, correct = 1, 0

    X1, X2, y = get_dataset(get_test_data(), flow_dict)
    m, n, p = X1.shape
    for i in range(m):
        X1_r = X1[i].reshape([1, n, p])
        target = predict_sequence(infenc, infdec, X1_r, 5, n_features, reverse_flow_dict)
        print(target)
    #     if array_equal(one_hot_decode(y[0],reverse_flow_dict), one_hot_decode(target,reverse_flow_dict)):
    #         correct += 1
    # print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
    # 查看预测结果
    # for _ in range(10):
    #     X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    #     target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    #     print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))


if __name__ == "__main__":
    # train_model()

    batch_size, timesteps, input_dim = None, 20, 1
    size = 1000
    import numpy as np

    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    print(x_train)
    print(y_train)
