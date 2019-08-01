import datetime

import torch as t
from model import TextRNN
from textclassification import data_processing, split_train_test
from torch import nn

embed_size = 128
hidden_size = 50
num_layers = 1
num_epochs = 500
batch_size = 2000

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def main():
    reviews_ints, labels, features, word_int_dict = data_processing(300)
    train_data, test_data, train_label, test_label = split_train_test(features, labels, 0.1)
    textrnn = TextRNN(300 * len(train_data), embed_size, hidden_size, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(textrnn.parameters(), lr=0.01)
    process_bar = len(train_data) // batch_size + 1
    # print('process_bar:', process_bar)
    for epoch in range(num_epochs):
        # h0 = [num_layers(1) * num_directions(1), batch_size, hidden_size]
        # h0 = h0.to(device)# 1*200*256
        # print(type(h0))
        for i in range(process_bar):
            x = train_data[batch_size * i:batch_size * (i + 1)]
            y = train_label[batch_size * i:batch_size * (i + 1)]
            # x = [batch_size * seq_length]
            x = t.LongTensor(x)
            y = t.LongTensor(y)
            # 下面一步中的输入x=[batch_size, seq_length, embed_size],
            # h0 = [batch_size, num_layers(1) * num_directions(1), hidden_size]
            # 输出output= [batch_size, seq_length, output_dim(num_directions * hidden_size)],
            # ht = [batch_size, num_layers * num_directions, hidden_size]
            output = textrnn(x)
            # print(output.size())
            # print(y.size())
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(str(datetime.datetime.now()) + '||epoch ' + str(epoch + 1) + '||step ' + str(
                i + 1) + ' | loss is: ' + str(loss.item()))

            if i % 5 == 0:
                # h0 = t.zeros(num_layers, len(test_data), hidden_size)
                test = t.LongTensor(test_data)
                # test = test.transpose(0, 1)
                # test_label = t.LongTensor(test_label)
                output = textrnn(test)
                pre_y = t.max(output,dim=1)[1].data.numpy().squeeze()
                print(len(pre_y))
                acc = sum(pre_y == test_label) / len(test_label)
                print('acc:', acc)


if __name__ == '__main__':
    main()
