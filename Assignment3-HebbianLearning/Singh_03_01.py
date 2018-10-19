# Singh, Saurabh
# 1001-568-347
# 2018-10-08
# Assignment-03-01

import sys
import Singh_03_02 as Heb
import Singh_03_03 as Act

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
    from tkinter import messagebox
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import random
import os

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

class LeftFrame:
    # This class creates and controls the widgets and figures in the left frame which
	# are used to display the activation functions.
    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.image_list = []
        self.data_folder = './Data'
        self.maxW = -0.001
        self.minW = 0.001
        self.min_alpha = 0.001
        self.max_alpha = 1.0
        self.learning_rate = 0.1
        self.train_data_percentage = 0.80
        self.no_of_classes = 10
        self.image_resolution = 784
        self.weight_and_bias = []
        self.training_data_index = []
        self.test_data_index = []
        self.target = []
        self.error_rate = []
        self.epoch_size = 100
        self.activation_type = "Linear"
        self.learning_method = "Delta Rule"

        self.read_images()
        self.initialize_weight_and_bias()
        self.divide_data()
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=2, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1)
        self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.10, 0.10, 0.85, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel('epoch')
        self.axes.set_ylabel('error_rate (in %)')
        self.axes.set_title("Error-Rate Graph")
        plt.ylim(0, 100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders
        #########################################################################

        # Weight w1 slider
        self.label_for_alpha = tk.Label(self.controls_frame, text="alpha (learning rate):", justify="center")
        self.label_for_alpha.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=self.min_alpha, to_=self.max_alpha, resolution=self.min_alpha,
                                            command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.learning_rate)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################

        self.label_for_learning_method = tk.Label(self.controls_frame, text="Select Learning Method",
                                                      justify="center")
        self.label_for_learning_method.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.learning_method_variable = tk.StringVar()
        self.learning_method_dropdown = tk.OptionMenu(self.controls_frame, self.learning_method_variable,
                                                          "Filtered Learning (Smoothing)", "Delta Rule",
                                                          "Unsupervised Hebb", command=lambda
                event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set(self.learning_method)
        self.learning_method_dropdown.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.label_for_activation_function = tk.Label(self.controls_frame, text="Transfer Functions",
                                                      justify="center")
        self.label_for_activation_function.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent",
                                                          "Linear",command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set(self.activation_type)
        self.activation_function_dropdown.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################

        self.learn_button = tk.Button(self.controls_frame, text="Adjust Weights (Learn)", width=25,
                                     command=self.train_callback)
        self.randomize_weights_button = tk.Button(self.controls_frame, text="Randomize Weights", width=25,
                                        command=self.initialize_weight_and_bias)
        self.confusion_button = tk.Button(self.controls_frame, text="Display Confusion Matrix", width=25,
                                            command=self.display_confusion_matrix_callback)
        self.learn_button.grid(row=0, column=3)
        self.randomize_weights_button.grid(row=1, column=3)
        self.confusion_button.grid(row=2, column=3)

    def alpha_slider_callback(self):
        self.learning_rate = np.float(self.alpha_slider.get())

    def learning_method_dropdown_callback(self):
        self.learning_method = self.learning_method_variable.get()

    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()

    def read_images(self):
        for file_name in os.listdir(self.data_folder):
            self.image_list.append(self.read_one_image_and_convert_to_vector(os.path.join(self.data_folder, file_name)))
            # Create a 10x1 target vector with the target being the first char of image filename
            target_vector = np.zeros((self.no_of_classes, 1))
            target_vector[int(file_name[0])] = 1
            # Append the target vector to the instance list "target"
            self.target.append(target_vector)

    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        img = img / 127.5 - 1
        img = np.append(img, 1)  # Append a 1 to vector for bias
        return img.reshape(-1, 1)  # reshape to column vector and return it

    def initialize_weight_and_bias(self):
        # Creating a 10 by (784+1) matrix to include bias
        self.weight_and_bias = np.random.uniform(self.minW, self.maxW, size=(self.no_of_classes, self.image_resolution+1))

    def divide_data(self):
        # Divide data into train and test
        train_data_count = int(self.train_data_percentage * len(self.image_list))
        image_index = list(range(0, len(self.image_list)))
        # Select "train_data_count" number of training data from index list "image_index" containing index for all images
        # And store it in "training_data_index". Those not present in this list is test data
        self.training_data_index = random.sample(image_index, train_data_count)
        self.test_data_index = [i for i in image_index if i not in self.training_data_index]

    def train_callback(self):
        for i in range(0,self.epoch_size):
            self.train()
            self.test()
            self.display_error_rate()

    def train(self):
        for j in self.training_data_index:
            output_vector = Act.calculate_activation(self.weight_and_bias, self.image_list[j],
                                                                          self.activation_type)
            self.weight_and_bias = Heb.calculate_new_weight_using_hebbian(self.weight_and_bias, self.learning_rate,
                                                self.target[j], output_vector, self.image_list[j],self.learning_method)

    def test(self, for_confusion_matrix=False):
        prediction_for_confusion_matrix = []
        true_for_confusion_matrix = []
        no_of_errors = 0
        for k in self.test_data_index:
            actual_vector = Act.calculate_activation(self.weight_and_bias, self.image_list[k],
                                                             self.activation_type)
            actual = np.argmax(actual_vector)
            target = np.argmax(self.target[k])
            if for_confusion_matrix:
                prediction_for_confusion_matrix.append(actual)
                true_for_confusion_matrix.append(target)
            if actual != target:
                no_of_errors = no_of_errors + 1

        if for_confusion_matrix:
            return (true_for_confusion_matrix, prediction_for_confusion_matrix)
        error_percent = no_of_errors / len(self.test_data_index) * 100
        # print("test_data: -> no_of_errors: ", no_of_errors, "percent: ", error_percent)
        self.error_rate.append(error_percent)

    def display_error_rate(self):
        self.axes.cla()
        self.axes.set_xlabel('epoch')
        self.axes.set_ylabel('error_rate (in %)')
        no_of_epochs = len(self.error_rate)
        plt.xlim(0, no_of_epochs)
        plt.ylim(0, 100)
        self.axes.plot(np.linspace(1, no_of_epochs, no_of_epochs), self.error_rate, 'b-', markersize=1)
        self.axes.xaxis.set_visible(True)
        self.axes.set_title("Error-Rate Graph")
        self.canvas.draw()

    def display_confusion_matrix_callback(self):
        y_true, y_pred = self.test(True)
        # print(confusion_matrix(y_true, y_pred))
        self.display_numpy_array_as_table(confusion_matrix(y_true, y_pred))

    def display_numpy_array_as_table(self, input_array):
        # This function displays a 1d or 2d numpy array (matrix).
        if input_array.ndim == 1:
            num_of_columns, = input_array.shape
            temp_matrix = input_array.reshape((1, num_of_columns))
        elif input_array.ndim > 2:
            print("Input matrix dimension is greater than 2. Can not display as table")
            return
        else:
            temp_matrix = input_array

        number_of_rows, num_of_columns = temp_matrix.shape
        fig, ax = plt.subplots()
        tb = plt.table(cellText=np.round(temp_matrix, 2), loc=(0, 0), cellLoc='center')
        for cell in tb.properties()['child_artists']:
            cell.set_height(1 / number_of_rows)
            cell.set_width(1 / num_of_columns)

        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.title('Assignment_03 --  Singh')
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()