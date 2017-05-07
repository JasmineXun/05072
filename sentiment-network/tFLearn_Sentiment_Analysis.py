# -*- coding:utf-8 -*-

"""
Author:xunying/Jasmine
Data:17-4-27
Time:下午8:33
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn as tfl
from tflearn.data_utils import to_categorical

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
from collections import Counter

total_counts = Counter()
for _, row in reviews.iterrows():  # 下划线是索引值
    total_counts.update(row[0].split(" "))  # row[0]得到每条评论,update()计数器统计每个单词出现次数并更新
print("Total words in data set: ", len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])  # 出现频率最高的前６０个单词
print(vocab[-1], ': ', total_counts[vocab[-1]])

word2idx = {word: i for i, word in enumerate(vocab)}  # 将单词作为key,对应的索引作为值


# 文本转化为向量
def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)


# Training data and Validation data and Test data
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])
Y = (labels == 'positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):]
trainX, trainY = word_vectors[train_split, :], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split, :], to_categorical(Y.values[test_split], 2)


# test_v=text_to_vector("my name is xun ying , I like food very much.")
# Network building
def build_model():
    tf.reset_default_graph()

    # inputs
    net = tfl.input_data([None, 10000])

    # hidden_layers
    net = tfl.fully_connected(net, 200, activation='ReLU')
    net = tfl.fully_connected(net, 25, activation="ReLU")

    # outputs
    net = tfl.fully_connected(net, 2, activation="softmax")
    net = tfl.regression(net, optimizer="sqd", learning_rate=0.1, loss="categoriccal_crossentropy")

    model = tfl.DNN(net)
    return model


# training the network
model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=50)

# test the network
predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:, 0], axis=0)
print("Test accuracy: ", test_accuracy)


# predict your sentence
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')

sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)