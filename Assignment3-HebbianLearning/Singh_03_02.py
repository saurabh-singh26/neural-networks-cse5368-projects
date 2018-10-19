# Singh, Saurabh
# 1001-568-347
# 2018-10-08
# Assignment-03-02

import numpy as np
# This module learns the new weight using variations of Hebb rule
def calculate_new_weight_using_hebbian(weight_old, learning_rate, target, actual, input, type):
    input_transpose = np.transpose(input)
    if type == 'Filtered Learning (Smoothing)':
        weight_new = (1 - learning_rate) * weight_old + learning_rate * np.dot(target, input_transpose)
    elif type == 'Delta Rule':
        weight_new = weight_old + learning_rate * np.dot((target - actual), input_transpose)
    elif type == 'Unsupervised Hebb':
        weight_new = weight_old + learning_rate * np.dot(actual, input_transpose)
    return weight_new