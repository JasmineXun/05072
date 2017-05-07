# -*- coding:utf-8 -*-

"""
Author:xunying/Jasmine
Data:17-4-5
Time:下午5:10
预测共享单车日使用量
instant      dteday  season  yr  mnth  hr  holiday  weekday  workingday
weathersit  temp   atemp   hum  windspeed  casual  registered  cnt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print(rides.head(20))
# print(rides[:24 * 10])
# plt.plot(rides[:24 * 10]['hr'],rides[:24 * 10]['cnt'])
# plt.show()
# 按照下面特征对应取值增加字段
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)  # 按照each作为前缀后加对应值命名
    rides = pd.concat([rides, dummies, ], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
# print(data.head())

# 计算下面这些特征对应的均值和标准差
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

# 将数据集划分成测试集、训练集和验证集
test_data = data[-21 * 24:]  # 后面２１天的数据作为测试数据

data = data[:-21 * 24]  # 移除测试集的数据
# 分离目标和特征
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# 将训练集分成两部分：training and validating(60天验证)
training_features, training_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设置输入层、隐藏层、输出层神经元数量
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 初始化权重
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5, (self.input_nodes,
                                                                                        self.hidden_nodes))
        self.weight_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes,
                                                                                         self.output_nodes))
        self.lr = learning_rate

        # 设置sigmoid函数为激活函数
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """
        训练网络　on batch of features and targets
        :param features: 二维数组，每行表示一条数据记录，每列表示一个特征
        :param targets: 一维数组，目标
        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_wiights_h_o = np.zeros(self.weight_hidden_to_output.shape)
        for X, y in zip(features, targets):
            ###Forward pass###
            # 隐藏层
            hidden_inputs = np.dot(np.array([X]), self.weights_input_to_hidden)  # 1*n n*m   = 1*m
            hidden_outputs = self.activation_function(hidden_inputs)
            # 输出层
            final_inputs = np.dot(hidden_outputs, self.weight_hidden_to_output)  # 1*m m*o =1*o
            final_outputs = self.activation_function(final_inputs)

            ###Backward pass###
            # 输出error
            # error = (targets - final_outputs) * final_outputs * (1 - final_outputs)

            # 计算隐藏层的error贡献值
            # hidden_error = error * hidden_outputs * (1 - hidden_outputs) * self.weight_hidden_to_output

            # backpropagated error
            output_error_team = (y - final_outputs) * final_outputs * (1 - final_outputs)
            hidden_error_team = np.dot(
                np.multiply(np.multiply(hidden_outputs, (1 - hidden_outputs)), self.weight_hidden_to_output.T).T,
                output_error_team)  # 2*1

            # Weight step(input to weight)
            delta_weights_i_h += np.dot(np.array([X]).T, hidden_error_team.T)  # n*m
            # Weight step(hidden to output)
            delta_wiights_h_o += np.dot(hidden_outputs.T, output_error_team)  # m*o

        # 修改权重
        self.weights_input_to_hidden += learning_rate * delta_weights_i_h
        self.weight_hidden_to_output += learning_rate * delta_wiights_h_o

    def run(self, features):
        """
        通过输入特征，执行前馈网络
        :param features: 一维特征值数组
        """
        # 隐藏层
        hidden_inputs = features
        hidden_outputs = np.dot(hidden_inputs, self.weights_input_to_hidden)  # k*n n*m =k*m

        # 输出层
        final_inputs = self.activation_function(hidden_outputs)
        final_outputs = np.dot(final_inputs, self.weight_hidden_to_output)  # k*m m*1 =k*1

        return self.activation_function(final_outputs)


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


"""
#######################################################################################

# 测试单元
import unittest

inputs = np.array([0.5, -0.2 - 0.1])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.1], [0.4, 0.5], [-0.3, 0.2]])
test_w_h_o = np.array([[0.3], [-0.1]])


class TestMethods(unittest.TestCase):
    # load data
    def test_data_path(self):
        # 测试文件路径是否已更改
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    # Unit test for network functionlity
    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # 测试激励函数是sigmoid函数
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    def test_train(self):
        # 测试权重是否被正确更新
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h
        network.weight_hidden_to_output = test_w_h_o

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.37275328],
                                              [-0.013172939]])))
        self.assertTrue(np.allclose(network.weight_hidden_to_output,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # 测试正确率
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weight_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


suit = unittest.TestLoader().loadTestsFromModule(TestMethods)
unittest.TextTestRunner().run(suit)

"""
#######################################################################################
# 调参
import sys

iterations = 1000
learning_rate = 0.0001
hidden_nodes = 200
output_nodes = 1

N_i = training_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
train_loss = 100
val_loss = 100
ii = 0
losses = {'train': [], 'validation': []}
while val_loss > 0.2:
    ii += 1
    # for ii in range(iterations):
    # Go through a random batch od 128 records from the training data set
    batch = np.random.choice(training_features.index, size=128)
    X, y = training_features.ix[batch].values, training_targets.ix[batch]['cnt']
    network.train(X, y)

    # 输出训练过程
    train_loss = MSE(network.run(training_features).T, training_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    print("Progress:%2.1f...training loss:%s...validation loss%s" % (ii, str(
        train_loss)[:5], str(val_loss)[:5]))
    # print("Progress:%2.1f%%...training loss:%s...validation loss%s" % ((100 * ii / float(iterations)), str(
    # train_loss)[:5], str(val_loss)[:5]))

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label="Training loss")
plt.plot(losses['validation'], label="Validation loss")
plt.legend()
_ = plt.ylim()
plt.show()

### test
test_loss = MSE(network.run(test_features).T, test_targets['cnt'].values)
print("test_loss:%s" % test_loss)
