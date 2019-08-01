"""
model层的定义
@date 2019.7.31
"""
import torch as t
from torch import nn
from torch import optim


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """
        初始化网络层
        :param vocab_size: 语料的大小
        :param embed_size: embedding的维度
        :param hidden_size: 隐藏层（RNN）的大小，即维度
        :param num_layers: RNN的层数
        """
        super(TextRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        # RNN初始化[x的维度，隐层的维度，网络层数]
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, X):
        """
        前向传播函数
        :param X: 输入的X，批数据
        :param h0: 初始状态h0
        :return:output，ht
        """
        # x = [batch_size, seq_length, embedding_dim]
        x = self.embedding(X)
        # print(x.size())
        # print("x---", x.size())
        # output = [batch_size, seq_length, output_dim(num_directions * hidden_size)]
        output, ht = self.rnn(x)
        # print(output.size())
        # output = [batch_size, output_dim(num_directions * hidden_size)]
        output = output[:, -1, :]
        # print(output.size())
        # output = [batch_size, 2]
        output = self.linear(output)
        # print(output.size())
        # print(output.size())
        # output = self.sig(output)
        # print(output)
        return output
