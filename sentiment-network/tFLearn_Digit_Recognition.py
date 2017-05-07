# -*- coding:utf-8 -*-

"""
Author:xunying/Jasmine
Data:17-4-28
Time:上午10:35
"""

import numpy as np
import tensorflow as tf
import tflearn

import tflearn.datasets.mnist as mnist

# training data and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# Visualizing the data
import matplotlib.pyplot as plt


# % matplotlib inline


# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28, 28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()


# Display the first (index 0) training image
# show_digit(1)


# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    # input layer
    net = tflearn.input_data([None, trainX.shape[1]])

    # hidden layer(s)
    net = tflearn.fully_connected(net, 128, activation='ReLU')
    net = tflearn.fully_connected(net, 32, activation='ReLU')

    # set how you want to train the model
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

    # This model assumes that your network is named "net"
    model = tflearn.DNN(net)
    return model


# Training
model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=100)

# Compare the labels that our model predicts with the actual labels

# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = np.array(model.predict(testX)).argmax(axis=1)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)
