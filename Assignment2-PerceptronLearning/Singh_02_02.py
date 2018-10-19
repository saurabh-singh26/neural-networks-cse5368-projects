# Singh, Saurabh
# 1001-568-347
# 2018-09-23
# Assignment-02-02

import numpy as np
# This module calculates the activation function
def calculate_activation_function(w1, w2, bias, p1, p2, type='Symmetrical Hard limit'):
    net_value = w1*p1 + w2*p2 + bias
    if type == 'Symmetrical Hard limit':
        net_value[net_value<0] = -1
        net_value[net_value>0] = +1
        activation = net_value
    elif type == "Linear":
        activation = net_value
    elif type == "Hyperbolic Tangent":
        activation = np.tanh(net_value)
    return activation