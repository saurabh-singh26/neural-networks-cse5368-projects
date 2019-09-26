# Singh, Saurabh
# 1001-568-347
# 2018-10-29
# Assignment-04-01

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
    from tkinter import messagebox
import matplotlib
import math

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class MainWindow(tk.Tk):
    # This class creates and controls the main window frames and widgets
    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.master_frame.rowconfigure(0, weight=10, minsize=400, uniform='xx')
        self.master_frame.rowconfigure(1, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=800)
        # create all the widgets
        self.left_frame = tk.Frame(self.master_frame)
        # Arrange the widget
        self.left_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
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

        self.file_name = 'data_set_1.csv'
        self.no_of_delayed_elements = 10
        self.learning_rate = 0.1
        self.train_data_percentage = 80
        self.stride = 1
        self.epoch_size = 10

        self.read_data()
        self.zero_weight_button_callback()
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1, minsize=200)

        # Plot for displaying Mean Square error (MSE)
        self.plot_frame = tk.Frame(self.master, borderwidth=2, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure = plt.figure(num=1)
        self.axes = self.figure.add_axes([0.10, 0.10, 0.85, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Iteration Number')
        self.axes.set_ylabel('Mean Square Error (MSE)')
        self.axes.set_title("MSE Graph for Price")
        plt.ylim(0, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # Plot for displaying Maximum Absolute error (MAE)
        self.plot_frame2 = tk.Frame(self.master, borderwidth=2, relief=tk.SUNKEN)
        self.plot_frame2.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure2 = plt.figure(num=2)
        self.axes2 = self.figure2.add_axes([0.10, 0.10, 0.85, 0.8])
        self.axes2 = self.figure2.gca()
        self.axes2.set_xlabel('Iteration Number')
        self.axes2.set_ylabel('Maximum Absolute Error (MAE)')
        self.axes2.set_title("MAE Graph for Price")
        plt.ylim(0, 2)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame2)
        self.plot_widget2 = self.canvas2.get_tk_widget()
        self.plot_widget2.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders
        #########################################################################

        # This slider selects number of delayed elements for each input (price change, volume change)
        # Range: 0 to 100; default: 10.
        self.no_of_delayed_elements_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_= 0, to_= 100, resolution = 1, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Number of Delayed Elements",
                                    command=lambda event: self.no_of_delayed_elements_slider_callback())
        self.no_of_delayed_elements_slider.set(self.no_of_delayed_elements)
        self.no_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_delayed_elements_slider_callback())
        self.no_of_delayed_elements_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Learning Rate: Adjust the learning rate(float).Range should be between 0.001 and 1.0.Default value = 0.1
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.001, to_=1.0, resolution=0.001, label="Learning Rate",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.learning_rate)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # This integer slider allows the user to select the percentage of the samples to be used for training.
        self.training_sample_size_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0, to_=100, resolution=1, label="Training Sample Size (Percentage)",
                                     command=lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.set(self.train_data_percentage)
        self.training_sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # This integer slider selects the step size for stride.Range 1 to 100. Default 1.
        self.stride_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                                    from_=1, to_=100, resolution=1, label="Stride",
                                                    command=lambda event: self.stride_slider_callback())
        self.stride_slider.set(self.stride)
        self.stride_slider.bind("<ButtonRelease-1>", lambda event: self.stride_slider_callback())
        self.stride_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        # This slider allows the user to change the number of times that the system goes over all the training samples
        self.no_of_iterations_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=1, to_=100, resolution=1, label="Number of Iterations",
                                      command=lambda event: self.no_of_iterations_slider_callback())
        self.no_of_iterations_slider.set(self.epoch_size)
        self.no_of_iterations_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_iterations_slider_callback())
        self.no_of_iterations_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################

        self.zero_weight_button = tk.Button(self.controls_frame, text="Set Weights to Zero", width=25,
                                     command=self.zero_weight_button_callback)
        self.adjust_LMS_weights_button = tk.Button(self.controls_frame, text="Adjust Weights (LMS)", width=25,
                                        command=self.adjust_LMS_weights_button_callback)
        self.adjust_direct_weights_button = tk.Button(self.controls_frame, text="Adjust Weights (Direct)", width=25,
                                            command=self.adjust_direct_weights_button_callback)
        self.zero_weight_button.grid(row=1, column=1)
        self.adjust_LMS_weights_button.grid(row=1, column=2)
        self.adjust_direct_weights_button.grid(row=1, column=3)

    def no_of_delayed_elements_slider_callback(self):
        self.no_of_delayed_elements = np.int(self.no_of_delayed_elements_slider.get()) #np.int?????

    def alpha_slider_callback(self):
        self.learning_rate = np.float(self.alpha_slider.get())

    def training_sample_size_slider_callback(self):
        self.train_data_percentage = np.int(self.training_sample_size_slider.get())

    def stride_slider_callback(self):
        self.stride = np.int(self.stride_slider.get())

    def no_of_iterations_slider_callback(self):
        self.epoch_size = np.int(self.no_of_iterations_slider.get())

    def zero_weight_button_callback(self):
        # print("zero_weight_button_callback")
        self.weight = np.zeros((self.get_input_size(), ))

    def get_input_size(self):
        return 2*(self.no_of_delayed_elements+1) + 1

    def adjust_LMS_weights_button_callback(self):
        self.zero_weight_button_callback()
        train_data = np.int(self.price_data.shape[0] * self.train_data_percentage/100)
        price_train_data = self.price_data[0 : train_data]
        volume_train_data = self.volume_data[0: train_data]
        price_test_data = self.price_data[train_data :]
        volume_test_data = self.volume_data[train_data :]
        mse_vector = []
        mae_vector = []
        (train_iteration_count, valid_index) = self.get_iteration_count(len(price_train_data))
        (test_iteration_count, valid_test_index) = self.get_iteration_count(len(price_test_data))
        # print("TIC: ", train_iteration_count)
        # print("valid_index: ", valid_index)
        for i in range(0, self.epoch_size):
            # Train
            for j in range(0, valid_index, self.stride):
                # print("Train Iteration ======================= ", j)
                input_vector = self.get_input_vector(j, price_train_data, volume_train_data)
                # print("input: ", input_vector)
                error = self.get_error(j, price_train_data, input_vector)
                self.weight = self.weight + 2*self.learning_rate*error*input_vector
                # print("weight: ", self.weight)

            # Test
            error_vector = []
            # print("LMS Weight: ", self.weight)
            for k in range(0, valid_test_index, self.stride):
                # print("Test Iteration ======================= ", k)
                input_vector = self.get_input_vector(k, price_test_data, volume_test_data)
                # print("input: ", input_vector)
                error_vector.append(self.get_error(k, price_test_data, input_vector))
                # print("error vector: ", error_vector)

            mse_vector.append(self.calculate_MSE(error_vector))
            mae_vector.append(self.calculate_MAE(error_vector))

        print("LMS Weight: ", self.weight)
        self.display_MSE(mse_vector)
        self.display_MAE(mae_vector)

    def get_iteration_count(self, data_size):
        count = 0
        start = 0
        while start + self.no_of_delayed_elements < data_size-2:
            count += 1
            start += self.stride
        return (count, start-self.no_of_delayed_elements)
        # return data_size - (self.no_of_delayed_elements + 1)

    def get_error(self, start_index, price, input):
        actual = np.dot(input, self.weight)
        target = price[start_index + self.no_of_delayed_elements + 1]
        error = target - actual
        # print("actual ", actual)
        # print("target ", target)
        # print("error ", error)
        return error

    def get_input_vector(self, start_index, price, volume):
        input = np.append(price[start_index: start_index + self.no_of_delayed_elements + 1][::-1],
                          volume[start_index: start_index + self.no_of_delayed_elements + 1][::-1])
        input = np.append(input, 1)  # Appending 1 for bias
        return input

    def calculate_MSE(self, error):
        mse = 0
        for i in error:
            mse += i * i
        mse /= len(error)
        # print("mse: ", mse)
        return mse

    def calculate_MAE(self, error):
        mae = 0
        for i in error:
            if (np.abs(i) > mae):
                mae = np.abs(i)
        # print("mae: ", mae)
        return mae

    def display_MSE(self, mse_error):
        self.axes.cla()
        self.axes.set_xlabel('Iteration Number')
        self.axes.set_ylabel('Mean Square Error (MSE)')
        self.axes.set_title("MSE Graph for Price")
        # print("MSE while display: ", mse_error)
        self.axes.plot(mse_error, 'b-', markersize=1)
        self.axes.xaxis.set_visible(True)
        self.canvas.draw()

    def display_MAE(self, mae_error):
        self.axes2.cla()
        self.axes2.set_xlabel('Iteration Number')
        self.axes2.set_ylabel('Maximum Absolute Error (MAE)')
        self.axes2.set_title("MAE Graph for Price")
        # print("MAE while display: ", mae_error)
        self.axes2.plot(mae_error, 'b-', markersize=1)
        self.axes2.xaxis.set_visible(True)
        self.canvas2.draw()

    def adjust_direct_weights_button_callback(self):
        # print("adjust_direct_weights_button_callback")
        train_data = np.int(self.price_data.shape[0] * self.train_data_percentage / 100)
        price_train_data = self.price_data[0: train_data]
        volume_train_data = self.volume_data[0: train_data]
        price_test_data = self.price_data[train_data:]
        volume_test_data = self.volume_data[train_data:]
        (train_iteration_count, valid_index) = self.get_iteration_count(len(price_train_data))
        (test_iteration_count, valid_test_index) = self.get_iteration_count(len(price_test_data))
        # train_iterations = self.get_iteration_count(price_train_data.size)
        h = np.zeros((self.get_input_size(), 1))
        R = np.zeros((self.get_input_size(), self.get_input_size()))

        # Calculate h and R from training set
        for j in range(0, valid_index, self.stride):
            input_vector = self.get_input_vector(j, price_train_data, volume_train_data)
            target = price_train_data[j + self.no_of_delayed_elements + 1]
            h += target*np.transpose([input_vector])
            R += np.matmul(np.transpose([input_vector]), [input_vector])

        h /= train_iteration_count
        # print("h: ", h)
        R /= train_iteration_count
        R = np.round(R, 5)
        w,v = np.linalg.eig(R)
        # print("Eigen values: ", w)
        print("Max eigen values: ", np.max(w))
        print("Alpha should be less than: ", 1/np.max(w))
        self.weight = np.matmul(np.linalg.inv(R), h)
        print("Direct weight: ", self.weight)
        # Test

        error_vector = []
        for k in range(0, valid_test_index, self.stride):
            input_vector = self.get_input_vector(k, price_test_data, volume_test_data)
            error_vector.append(self.get_error(k, price_test_data, input_vector))

        mse = [self.calculate_MSE(error_vector)]
        mae = [self.calculate_MAE(error_vector)]
        print("Minimum MSE: ", mse)
        print("Minimum MAE: ", mae)
        x = np.linspace(0, self.epoch_size, self.epoch_size)
        self.axes2.plot(x, mae*x.size, 'r-', markersize=1)
        self.axes.plot(x, mse*x.size, 'r-', markersize=1)
        self.canvas.draw()
        self.canvas2.draw()
        # self.display_MSE(mse)
        # self.display_MAE(mae)

    def read_data(self):
        data = np.loadtxt(self.file_name, skiprows=1, delimiter=',', dtype=np.float32)
        price = data[:, 0]
        volume = data[:, 1]
        self.price_data = self.normalize(price)
        self.volume_data = self.normalize(volume)
        # print("Data Price: ", self.price_data)
        # print("Data Volume: ", self.volume_data)

    def normalize(self, data):
        max = 0
        for i in data:
            if(math.fabs(i)>max):
                max = math.fabs(i)
        data /= max
        return data

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.title('Assignment_04 --  Singh')
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()