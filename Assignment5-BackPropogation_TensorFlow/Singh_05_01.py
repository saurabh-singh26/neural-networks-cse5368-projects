# Singh, Saurabh
# 1001-568-347
# 2018-11-26
# Assignment-05-01

# References:
# 1) https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
# 2) http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# 3) https://www.tensorflow.org/

import sys
import Singh_05_02 as GenData

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
    from tkinter import messagebox
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class MainWindow(tk.Tk):
    # This class creates and controls the main window frames and widgets
    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.master_frame.rowconfigure(0, weight=10, minsize=400, uniform='xx')
        self.master_frame.rowconfigure(1, weight=1, minsize=10, uniform='xx')
        # create all the widgets
        self.left_frame = tk.Frame(self.master_frame)
        # Arrange the widget
        self.left_frame.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)
        self.display_activation_functions.generate_data()

class LeftFrame:
    # This class creates and controls the widgets and figures in the left frame which
	# are used to display the activation functions.
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root

        #########################################################################
        #  Set up the constants and default values
        #########################################################################

        self.lambda_val = 0.1
        self.learning_rate = 0.01
        self.no_of_nodes_in_hidden_layer = 100
        self.no_of_samples = 200
        self.no_of_classes = 4
        self.epoch = 10
        self.activation_type = "Relu"
        self.data_type = "s_curve"

        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################

        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=2, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1)
        self.figure = plt.figure("")
        self.a = self.figure.add_subplot(111)
        self.a.set_xlabel("X")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders
        #########################################################################

        # "Lambda": (Weight regularization). Range should be between 0.0 and 1.0. Default value = 0.01 Increments=0.01
        self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD", activebackground="#FF0000",
                                      highlightcolor="#00FFFF", label="Lambda",
                                      command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_val)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # "Num. of Nodes in Hidden Layer": Range 1 to 500. Default value=100  increment=1
        self.no_of_nodes_in_hidden_layer_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(),
                                                           orient=tk.HORIZONTAL, from_=1, to_=500, resolution=1,
                                                           label="Num. of Nodes in Hidden Layer",
                                                           command=lambda
                                                               event: self.no_of_nodes_in_hidden_layer_slider_callback())
        self.no_of_nodes_in_hidden_layer_slider.set(self.no_of_nodes_in_hidden_layer)
        self.no_of_nodes_in_hidden_layer_slider.bind("<ButtonRelease-1>",
                                                     lambda event: self.no_of_nodes_in_hidden_layer_slider_callback())
        self.no_of_nodes_in_hidden_layer_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # "Alpha": (Learning rate) Range should be between 0.000 and 1.0. Default value = 0.1 increments=.001
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.0, to_=1.0, resolution=0.01, label="Alpha",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.learning_rate)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # Number of epochs
        self.epoch_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=10, to_=1000, resolution=1, label="Epoch",
                                     command=lambda event: self.epoch_slider_callback())
        self.epoch_slider.set(self.epoch)
        self.epoch_slider.bind("<ButtonRelease-1>", lambda event: self.epoch_slider_callback())
        self.epoch_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        # This slider determines the number of samples which will be generated for input data.
        # Range 4 to 1000. Default value=200, increment=1.
        self.no_of_samples_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=4, to_=1000, resolution=1, label="Number of Samples",
                                             command=lambda event: self.no_of_samples_slider_callback())
        self.no_of_samples_slider.set(self.no_of_samples)
        self.no_of_samples_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_samples_slider_callback())
        self.no_of_samples_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # This slider determines the number of classes which will be generated for input data.
        # Range 2 to 10. Default value=4 Increments=1
        self.no_of_classes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=2, to_=10, resolution=1, label="Number of Classes",
                                             command=lambda event: self.no_of_classes_slider_callback())
        self.no_of_classes_slider.set(self.no_of_classes)
        self.no_of_classes_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_classes_slider_callback())
        self.no_of_classes_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################

        self.adjust_weights_button = tk.Button(self.controls_frame, text="Adjust Weights (Train)", width=25,
                                               command=self.adjust_weights_button_callback)
        self.reset_weights_button = tk.Button(self.controls_frame, text="Reset Weights", width=25,
                                              command=self.reset_weights_button_callback)
        self.adjust_weights_button.grid(row=1, column=3)
        self.reset_weights_button.grid(row=2, column=3)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################

        # A drop-down to select between two transfer functions for the hidden layer (Relu, and Sigmoid). Default: Relu
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Hidden Layer Transfer Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Sigmoid", "Relu", command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set(self.activation_type)
        self.activation_function_dropdown.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # A drop-down box to allow the user to select between four different types of generated data
        self.data_type_label = tk.Label(self.controls_frame, text="Type of generated data", justify="center")
        self.data_type_label.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.data_type_variable = tk.StringVar()
        self.data_type_dropdown = tk.OptionMenu(self.controls_frame, self.data_type_variable,
                                                "s_curve", "blobs", "swiss_roll", "moons",
                                                command=lambda event: self.data_type_dropdown_callback())
        self.data_type_variable.set(self.data_type)
        self.data_type_dropdown.grid(row=2, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

    def lambda_slider_callback(self):
        self.lambda_val = np.int(self.lambda_slider.get())

    def alpha_slider_callback(self):
        self.learning_rate = np.float(self.alpha_slider.get())

    def no_of_nodes_in_hidden_layer_slider_callback(self):
        self.no_of_nodes_in_hidden_layer = np.int(self.no_of_nodes_in_hidden_layer_slider.get())

    def epoch_slider_callback(self):
        self.epoch = np.int(self.epoch_slider.get())

    def no_of_samples_slider_callback(self):
        self.no_of_samples = np.int(self.no_of_samples_slider.get())
        self.generate_data()

    def no_of_classes_slider_callback(self):
        self.no_of_classes = np.int(self.no_of_classes_slider.get())
        self.generate_data()

    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()

    def data_type_dropdown_callback(self):
        self.data_type = self.data_type_variable.get()
        self.generate_data()

    def adjust_weights_button_callback(self):
        self.train()

    def reset_weights_button_callback(self):
        low = -0.001
        high = 0.001
        self.weight_hidden_layer = np.random.uniform(low, high, size=(2, self.no_of_nodes_in_hidden_layer))
        self.bias_hidden_layer = np.random.uniform(low, high, size=(1, self.no_of_nodes_in_hidden_layer))
        self.weight_output_layer = np.random.uniform(low, high, size=(self.no_of_nodes_in_hidden_layer, self.no_of_classes))
        self.bias_output_layer = np.random.uniform(low, high, size=(1, self.no_of_classes))

    def generate_data(self):
        self.input, self.class_label = GenData.generate_data(self.data_type, self.no_of_samples, self.no_of_classes)
        self.plot_data()
        self.reset_weights_button_callback()

    def plot_data(self):
        self.a.cla()
        self.a.scatter(self.input[:, 0], self.input[:, 1], c=self.class_label, cmap=plt.cm.Accent)
        self.canvas.draw()

    def get_class_label_as_vector(self):
        out = []
        for i in range(len(self.class_label)):
            temp = [0.0]*self.no_of_classes
            temp[self.class_label[i]] = 1
            out.append(temp)
        return out

    def train(self):
        p = tf.constant(self.input)
        y = tf.constant(self.get_class_label_as_vector())
        w1 = tf.Variable(self.weight_hidden_layer)
        b1 = tf.Variable(self.bias_hidden_layer)
        w2 = tf.Variable(self.weight_output_layer)
        b2 = tf.Variable(self.bias_output_layer)

        hidden_out = tf.add(tf.matmul(p, w1), b1)
        if self.activation_type=="Relu":
            hidden_out = tf.nn.relu(hidden_out)
        if self.activation_type=="Sigmoid":
            hidden_out = tf.nn.sigmoid(hidden_out)

        regularizer = tf.cast(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2), tf.float32)
        output_out = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))
        output_out_clipped = tf.cast(tf.clip_by_value(output_out, 1e-10, 0.9999999), tf.float32)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(output_out_clipped)
                                                      + (1 - y) * tf.log(1 - output_out_clipped), axis=1))
        cross_entropy_with_L2 = tf.reduce_mean(cross_entropy + self.lambda_val * regularizer)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy_with_L2)
        init_op = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output_out, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.setUpMeshGridForBoundaryDisplay()
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(self.epoch):
                _, c, a, w1_a, b1_a, w2_a, b2_a = sess.run([optimizer, cross_entropy_with_L2, accuracy, w1, b1, w2, b2])
                # print("Epoch:", epoch+1, "accuracy:", a, "cross_entropy", c)
                self.display_boundary(w1_a, b1_a, w2_a, b2_a)

    def setUpMeshGridForBoundaryDisplay(self):
        x_axis_interval = self.a.get_xaxis().get_view_interval()
        y_axis_interval = self.a.get_yaxis().get_view_interval()
        x_axis_linspace = np.linspace(x_axis_interval[0], x_axis_interval[1], 100, endpoint=True)
        y_axis_linspace = np.linspace(y_axis_interval[0], y_axis_interval[1], 100, endpoint=True)
        self.xx, self.yy = np.meshgrid(x_axis_linspace, y_axis_linspace)
        xx_a = np.asarray(self.xx).reshape(-1)
        yy_a = np.asarray(self.yy).reshape(-1)
        self.p = []
        for x, y in zip(xx_a, yy_a):
            self.p.append([x, y])
        self.p = np.array(self.p)

    def display_boundary(self, w1, b1, w2, b2):
        activation_hidden = self.calculate_activation_function(w1, b1, self.p, self.activation_type)
        activation_output = self.calculate_activation_function(w2, b2, activation_hidden)
        activation_output = np.array([np.argmax(activation_output, axis=1)])
        activation_output = activation_output.reshape(self.xx.shape)
        self.a.pcolormesh(self.xx, self.yy, activation_output, cmap=plt.cm.Pastel1, zorder=0)
        self.canvas.draw()

    def calculate_activation_function(self, w, bias, p, atype="Linear"):
        net_value = np.add(np.matmul(p, w), bias)
        if atype == 'Relu':
            activation = net_value * (net_value > 0)
        elif atype == "Linear":
            activation = net_value
        elif atype == "Sigmoid":
            activation = 1.0 / (1 + np.exp(-net_value))
        return activation

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.title('Assignment_05 --  Singh')
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()