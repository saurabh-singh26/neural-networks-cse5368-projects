# Singh, Saurabh
# 1001-568-347
# 2018-10-08
# Assignment-03-03

import numpy as np
# This module calculates the activation function
def calculate_activation(weight, input, type):
    net_value = np.dot(np.asmatrix(weight), np.asmatrix(input))
    if type == 'Symmetrical Hard limit':
        net_value[net_value<0] = -1
        net_value[net_value>0] = +1
        activation = net_value
    elif type == "Linear":
        activation = softmax(net_value)
    elif type == "Hyperbolic Tangent":
        activation = np.tanh(net_value)
    return activation

def softmax(vector):
    sum = 0.0001
    for i in range(vector.size):
        vector[i] = np.exp(vector[i])
        sum += vector[i]
    return vector/sum