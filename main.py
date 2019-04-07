from random import random
from math import exp
import numpy
from matplotlib import pyplot
import sys
import os

'''
HELPER FUNCTIONS
'''
def target(row, theta, bias):
    # print('row: ', row)
    # print('theta: ', theta)
    # print('bias: ', bias)

    result = ((float(row[0]) * theta[0]) + (float(row[1]) * theta[1]) +
              (float(row[2]) * theta[2]) + (float(row[3]) * theta[3])) + float(bias)

    # print('target: ', result)

    return result

def clear(): return os.system('clear')

def sigmoid(target):
    target = -1 * target

    try:
        exp(target)
    except OverflowError:
        print('target: ', target)
        # target = round(target)

    result = float(1) / (1 + exp(target))

    print('sigmoid: ', result)

    return result

def s_hidden(s_outs, v_weights, o_hidden):
    # print('s_outs: ', s_outs)
    # print('v_weights: ', v_weights)
    # print('o_hidden: ', o_hidden)

    result = (((s_outs[0] * v_weights[0]) + (s_outs[1] * v_weights[1])) * (o_hidden) * (1 - o_hidden))

    # print('s_hidden: ', result)
    return result

def s_out(y, target):
    # print('y: ', y)
    # print('target: ', target)

    result = ((y - target) * y * (1 - y))
    # print('result: ', result)
    return result

def prediction(sigmoid):
    return round(sigmoid)

def error(prediction, actual):
    return ((prediction - actual) ** 2)

def get_category(name):
    categories = {
        'Iris-setosa': (0, 0),
        'Iris-versicolor': (0, 1),
        'Iris-virginica': (1, 0),
    }
    
    return categories[name]

def rearrange(data):
    arranged = []
    for (setosa, virginica, versicolor) in zip(data[:50], data[50:100], data[100:150]):
        arranged.append(setosa)
        arranged.append(virginica)
        arranged.append(versicolor)
    return arranged

def tao_out(tao_out, omega, y):
    return [s_hidden(tao_out, omega[0], y), s_hidden(tao_out, omega[1], y), s_hidden(tao_out, omega[2], y), s_hidden(tao_out, omega[3], y)]

data = open('iris_data.csv')

INPUT_VARIABLES = 4
HIDDEN_NODES = 4
OUTPUT_NODES = 2
LEARNING_RATE = 0.1
EPOCH = 300
TRAIN_LENGTH = 120
# thetas = ([[random()] * INPUT_VARIABLES] * HIDDEN_NODES)
thetas = ([[random() for i in range(INPUT_VARIABLES)] for j in range(HIDDEN_NODES)])
# thetas_bias = ([random()] * HIDDEN_NODES)
thetas_bias = ([random() for i in range(HIDDEN_NODES)])
# omegas = ([[random()] * HIDDEN_NODES] * OUTPUT_NODES)
omegas = ([[random() for i in range(HIDDEN_NODES)] for j in range(OUTPUT_NODES)])
# omegas_bias = ([random()] * OUTPUT_NODES)
omegas_bias = ([random() for i in range(OUTPUT_NODES)])

'''
DATA PRE-PROCESSING
'''
data = data.read().split('\n')
data = rearrange(data)

categories = ([get_category(row.split(',')[4]) for row in data])
data = ([row.split(',')[:4] for row in data])
data = [[float(col) for col in row] for row in data]

'''
TRAIN
'''
# row = data[0]
# category = categories[0]
for (row, category) in zip(data[:TRAIN_LENGTH], categories[:TRAIN_LENGTH]):
# for (row, category) in zip(data[:4], categories):
    y_hidden = []
    y_hidden = ([target(row, theta, bias) for (theta, bias) in zip(thetas, thetas_bias)])

    try:
        sigmoid_hidden = ([sigmoid(y) for y in y_hidden])
    except:
        print('row:', row)
        print('y_hidden:', y_hidden)

    y_out = ([target(sigmoid_hidden, omega, bias) for (omega, bias) in zip(omegas, omegas_bias)])

    try:
        sigmoid_out = ([sigmoid(y) for y in y_out])
    except:
        print('row:', row)
        print('y_out:', y_out)

    # IF THERE'S ERROR, CHECK HERE.
    tao_out = ([s_out(sigmoid, target) for (sigmoid, target) in zip(sigmoid_out, category)])
    
    # tao_hidden = ([s_hidden(tao_out, omega, y) for (omega, y) in zip(omegas.T, y_hidden)])
    tao_hidden = ([s_hidden(tao_out, [omegas[0][i], omegas[1][i]], y_hidden[i]) for i in range(4)])

    print('thetas:', thetas)
    print('sigmoid hidden:', sigmoid_hidden)
    print('tao hidden:', tao_hidden)
    thetas = [[(theta[0] + (LEARNING_RATE * (col * tao))), (theta[1] + (LEARNING_RATE * (col * tao))), (theta[2] + (LEARNING_RATE * (col * tao))), (theta[3] + (LEARNING_RATE * (col * tao)))] for (theta, col, tao) in zip(thetas, sigmoid_hidden, tao_hidden)]
    thetas_bias = [bias + (LEARNING_RATE * tao) for (bias, tao) in zip(thetas_bias, tao_hidden)]
    omegas = ([[(omega[0] + (LEARNING_RATE * (y * tao))), (omega[1] + (LEARNING_RATE * (y * tao))), (omega[2] + (LEARNING_RATE * (y * tao))), (omega[3] + (LEARNING_RATE * (y * tao)))] for (omega, y, tao) in zip(omegas, sigmoid_out, tao_out)])
    omegas_bias = [bias + (LEARNING_RATE * tao) for (bias, tao) in zip(omegas_bias, tao_out)]

    print('\n\n\n\n\n\n\n')

'''
TEST
'''
errors = []
error = 0
# row = test[0]
# category = categories[0]
clear()
for (row, category) in zip(data[TRAIN_LENGTH:], categories[TRAIN_LENGTH:]):
    y_hidden = [target(row, theta, bias) for (theta, bias) in zip(thetas, thetas_bias)]
    sigmoid_hidden = [sigmoid(y) for y in y_hidden]

    y_out = [target(sigmoid_hidden, omega, bias) for (omega, bias) in zip(omegas, omegas_bias)]
    sigmoid_out = [sigmoid(y) for y in y_out]

    error1 = ((sigmoid_out[0] - category[0])  ** 2)
    error2 = ((sigmoid_out[1] - category[1])  ** 2)
    error += ((error1 + error2) / 2)
    print('y_hidden:', y_hidden, ', sigmoid_hidden:', sigmoid_hidden,
          ', y_out:', y_out, ', sigmoid_out:', sigmoid_out, ', categories:', category, ', error 1:', error1, ', error 2:', error2)

    errors.append((error) / TRAIN_LENGTH)

pyplot.plot(errors)

pyplot.show()
