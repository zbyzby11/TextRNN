"""
数据的预处理部分，主要功能是去掉标点符号，reviews和labels到id的映射，训练集和测试集的划分
@date 2019.7.31
"""
import torch as t
import numpy as np
from torch import nn
from torch.optim import optimizer
from string import punctuation
from collections import Counter
from sklearn.model_selection import train_test_split


def data_processing(seq_len):
    """
    数据部分的预处理
    :param seq_len: 评论字数阈值，小于和大于这个阈值需要经过一定的处理
    :return: 每个评论的id列表和对应的情感列表和一个经过字数限制的评论列表,还有一个word字典
    """
    # 读取review.txt和label.txt中的内容
    with open('../data/reviews.txt', 'r') as f:
        reviews = f.read()  # 字符串
    with open('../data/labels.txt', 'r') as f:
        labels = f.read()  # 字符串

    # 在reviews中去掉标点符号
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews = all_text.split('\n')  # 列表，元素为去掉标点的review.txt中的每一行
    all_text = ' '.join(reviews)
    words = all_text.split()  # 列表，每个元素是一个word
    # 统计每个词出现的次数
    counter = Counter(words)
    word_sorted = sorted(counter, key=counter.get, reverse=True)  # 以出现次数从大到小排列word
    word_to_int = {word: num + 1 for num, word in enumerate(word_sorted)}
    # 将评论转化为数字id
    reviews_ints = []
    for review in reviews:
        reviews_ints.append([word_to_int[word] for word in review.split()])
    # 将标签转化为id，1代表positive，0代表negative
    labels = labels.split('\n')
    labels = np.array([1 if each == 'positive' else 0 for each in labels])
    # 删除长度为0的review和对应的label
    non_zero_index = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    reviews_ints = [reviews_ints[ii] for ii in non_zero_index]
    labels = np.array([labels[ii] for ii in non_zero_index])
    # 对于每一条评论，设定一个阈值，如果评论的字数少于这个阈值，那么从左边补上0，反之若大于阈值，则截取到这个阈值的字数
    features = np.array(
        [review[:seq_len] if len(review) > seq_len else [0] * (seq_len - len(review)) + review for review in
         reviews_ints])
    # print(features.shape)
    # print(len(reviews_ints))
    # print(len(labels))
    return reviews_ints, labels, features, word_to_int


def split_train_test(features, labels, split_rate):
    """
    进行训练集和测试集的划分
    :param features: 矩阵X，即处理过的reviews
    :param labels: y，对应的reivews的评论
    :param split_rate: 测试集占的总比例，float
    :return: 划分好的训练集合测试集
    """
    train_data, test_data, train_label, test_label = train_test_split(features, labels, test_size=split_rate)
    return train_data, test_data, train_label, test_label


def main():
    reviews_ints, labels, features, word2int = data_processing(300)
    train_data, test_data, train_label, test_label = split_train_test(features, labels, 0.1)
    print(len(train_data))
    print(len(train_label))
    print(len(test_data))
    print(len(test_label))
    print(train_data.shape)


if __name__ == '__main__':
    main()
