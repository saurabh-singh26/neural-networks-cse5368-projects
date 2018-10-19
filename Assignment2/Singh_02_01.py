# Singh, Saurabh
# 1001-568-347
# 2018-09-23
# Assignment-02-01

import Singh_02_02
import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
    from tkinter import messagebox
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.colors as c
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
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.weight_w1 = 1
        self.weight_w2 = 1
        self.bias = 0.0
        self.resolution = 100
        self.learning_rate = 0.1
        self.positive_samples = np.random.randint(self.xmin, self.xmax+1, size=(2, 2))
        self.negative_samples = np.random.randint(self.ymin, self.ymax+1, size=(2, 2))

        self.activation_type = "Symmetrical Hard limit"
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1)
        self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
        # self.axes = self.figure.add_axes()
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        # self.axes.margins(0.5)
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
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
        self.label_for_weight_w1 = tk.Label(self.controls_frame, text="Weight w1:", justify="center")
        self.label_for_weight_w1.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.weight_w1_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01,
                                            command=lambda event: self.weight_w1_slider_callback())
        self.weight_w1_slider.set(self.weight_w1)
        self.weight_w1_slider.bind("<ButtonRelease-1>", lambda event: self.weight_w1_slider_callback())
        self.weight_w1_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # Weight w2 slider
        self.label_for_weight_w2 = tk.Label(self.controls_frame, text="Weight w2:", justify="center")
        self.label_for_weight_w2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.weight_w2_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                         from_=-10.0, to_=10.0, resolution=0.01,
                                         command=lambda event: self.weight_w2_slider_callback())
        self.weight_w2_slider.set(self.weight_w2)
        self.weight_w2_slider.bind("<ButtonRelease-1>", lambda event: self.weight_w2_slider_callback())
        self.weight_w2_slider.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # Bias slider
        self.label_for_bias = tk.Label(self.controls_frame, text="Bias:", justify="center")
        self.label_for_bias.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.bias_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                         from_=-10.0, to_=10.0, resolution=0.01,
                                         command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################

        self.label_for_activation_function = tk.Label(self.controls_frame, text="Transfer Function:",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent",
                                                          "Linear",command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################

        self.train_button = tk.Button(self.controls_frame, text="Train", width=16,
                                     command=self.train_callback)
        self.random_data_button = tk.Button(self.controls_frame, text="Create Random Data", width=16,
                                        command=self.create_random_data_callback)
        self.train_button.grid(row=2, column=2)
        self.random_data_button.grid(row=2, column=3)

    def display_activation_function(self):
        p1_values = np.linspace(-10, 10, self.resolution, endpoint=True)
        p2_values = np.linspace(-10, 10, self.resolution, endpoint=True)
        xx, yy = np.meshgrid(p1_values, p2_values)
        activation = Singh_02_02.calculate_activation_function(self.weight_w1, self.weight_w2, self.bias, xx,
                                                               yy, self.activation_type)
        self.axes.cla()
        self.axes.set_xlabel('p1')
        self.axes.set_ylabel('p2')
        self.axes.plot(p1_values,-1*(self.bias+self.weight_w1*p1_values)/self.weight_w2, linestyle='-')
        self.axes.plot(self.positive_samples[:, 0], self.positive_samples[:,1], 'bx', markersize=6)
        self.axes.plot(self.negative_samples[:, 0], self.negative_samples[:,1], 'bo', markersize=6)
        self.axes.pcolormesh(p1_values, p2_values, activation, cmap=c.ListedColormap(['r','g']))
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.activation_type)
        self.canvas.draw()

    def weight_w1_slider_callback(self):
        self.weight_w1 = np.float(self.weight_w1_slider.get())
        self.display_activation_function()

    def weight_w2_slider_callback(self):
        self.weight_w2 = np.float(self.weight_w2_slider.get())
        self.display_activation_function()

    def bias_slider_callback(self):
        self.bias = np.float(self.bias_slider.get())
        self.display_activation_function()

    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()
        self.display_activation_function()

    def train_callback(self):
        xx_pos = self.positive_samples[:, 0]
        yy_pos = self.positive_samples[:, 1]
        xx_neg = self.negative_samples[:, 0]
        yy_neg = self.negative_samples[:, 1]
        for i in range(100):
            count = 0
            activation_pos = Singh_02_02.calculate_activation_function(self.weight_w1, self.weight_w2, self.bias,
                                                                       xx_pos, yy_pos, self.activation_type)
            activation_neg = Singh_02_02.calculate_activation_function(self.weight_w1, self.weight_w2, self.bias,
                                                                       xx_neg, yy_neg, self.activation_type)
            for j in range(2):
                error = 1 - activation_pos[j]
                self.weight_w1 = self.weight_w1 + self.learning_rate*error*xx_pos[j]
                self.weight_w2 = self.weight_w2 + self.learning_rate*error*yy_pos[j]
                self.bias = self.bias + error
                if error==0:
                    count = count+1
                else:
                    count = count-1
                self.display_activation_function()
            for j in range(2):
                error = -1 - activation_neg[j]
                self.weight_w1 = self.weight_w1 + self.learning_rate*error*xx_neg[j]
                self.weight_w2 = self.weight_w2 + self.learning_rate*error*yy_neg[j]
                self.bias = self.bias + error
                if error==0:
                    count = count+1
                else:
                    count = count-1
                self.display_activation_function()
            if count == 4:
                break

    def create_random_data_callback(self):
        self.positive_samples = np.random.randint(self.xmin, self.xmax + 1, size=(2, 2))
        self.negative_samples = np.random.randint(self.ymin, self.ymax + 1, size=(2, 2))
        self.display_activation_function()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()


main_window = MainWindow(debug_print_flag=False)
main_window.title('Assignment_02 --  Singh')
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()
