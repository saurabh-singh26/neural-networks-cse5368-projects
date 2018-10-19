# Singh, Saurabh
# 1001-568-347
# 2018-09-08
# Assignment-01-02

# References: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html
import numpy as np
# This module calculates the activation function
def calculate_activation_function(weight,bias,input_array,type='Sigmoid'):
	net_value = weight * input_array + bias
	if type == 'Sigmoid':
		activation = 1.0 / (1 + np.exp(-net_value))
	elif type == "Linear":
		activation = net_value
	elif type == "Hyperbolic Tangent":
		activation = np.tanh(net_value)
	elif type == "Positive linear (RELU)":
		activation = net_value * (net_value > 0)
	return activation