# Singh, Saurabh
# 1001-568-347
# 2018-10-29
# Assignment-06-01

# Reference
# https://keras.io/getting-started/sequential-model-guide/#vgg-like-convnet

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
    from tkinter import messagebox
import matplotlib
import os, fnmatch, math
import pickle

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import keras as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras import regularizers

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

        self.alpha = 0.1
        self.lambda_reg = 0.01
        self.F1 = 32
        self.K1 = 3
        self.F2 = 32
        self.K2 = 3
        self.train_data_percentage = 20
        self.epoch = 1
        self.batch_size = 32
        self.data_folder = './Data'
        self.train_data_count = 50000

        self.read_train_data()
        self.read_test_data()
        self.create_base_model()
        self.reset_weight_button_callback()
        self.print_confusion_matrix()
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1, minsize=200)

        # Plot for displaying error graph
        self.plot_frame = tk.Frame(self.master, borderwidth=2, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0.10, 0.10, 0.85, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Error rate (in percentage)')
        self.axes.set_title("Error Graph")
        plt.ylim(0, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders
        #########################################################################

        # "Alpha": (Learning rate) Range should be between 0.000 and 1.0. Default value = 0.1 increments=.001
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Alpha",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # "Lambda": (Weight regularization). Range should be between 0.0 and 1.0. Default value = 0.01 Increments=0.01
        self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_= 0.0, to_= 1.0, resolution = 0.01, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Lambda",
                                    command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_reg)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # "F1": Number of filters in the first layer. Range 1 to 64. Default value=32  increment=1
        self.F1_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=1, to_=64, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="F1",
                                     command=lambda event: self.F1_slider_callback())
        self.F1_slider.set(self.F1)
        self.F1_slider.bind("<ButtonRelease-1>", lambda event: self.F1_slider_callback())
        self.F1_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # "K1": Kernel size for the filters in the first layer. Range 3 to 7. Default value=3, increment=2.
        self.K1_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                                    from_=3, to_=7, resolution=2, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="K1",
                                                    command=lambda event: self.K1_slider_callback())
        self.K1_slider.set(self.K1)
        self.K1_slider.bind("<ButtonRelease-1>", lambda event: self.K1_slider_callback())
        self.K1_slider.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # "F2": Number of filters in the second layer. Range 1 to 64. Default value=32  increment=1
        self.F2_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                  from_=1, to_=64, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="F2",
                                  command=lambda event: self.F2_slider_callback())
        self.F2_slider.set(self.F1)
        self.F2_slider.bind("<ButtonRelease-1>", lambda event: self.F2_slider_callback())
        self.F2_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # "K2": Kernel size for the filters in the first layer. Range 3 to 7. Default value=3, increment=2.
        self.K2_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                  from_=3, to_=7, resolution=2, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="K2",
                                  command=lambda event: self.K2_slider_callback())
        self.K2_slider.set(self.K1)
        self.K2_slider.bind("<ButtonRelease-1>", lambda event: self.K2_slider_callback())
        self.K2_slider.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # "Training Sample Size (Percentage)": This integer slider allows the user to select the percentage of the
        # samples to be used for training. range 0% to 100%. Default value should be 20% which means the 20% of the
        # training samples, will be used for training.
        self.train_data_percentage_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0, to_=100, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Training Sample Size (Percentage)")
        self.train_data_percentage_slider.set(self.train_data_percentage)
        self.train_data_percentage_slider.bind("<ButtonRelease-1>", lambda event: self.train_data_percentage_slider_callback())
        self.train_data_percentage_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        # Epoch slider
        self.epoch_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                  from_=1, to_=10, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                  highlightcolor="#00FFFF", label="Epoch",
                                  command=lambda event: self.epoch_slider_callback())
        self.epoch_slider.set(self.epoch)
        self.epoch_slider.bind("<ButtonRelease-1>", lambda event: self.epoch_slider_callback())
        self.epoch_slider.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################

        self.reset_weight_button = tk.Button(self.controls_frame, text="Reset Weights", width=25,
                                     command=self.reset_weight_button_callback)
        self.adjust_weights_button = tk.Button(self.controls_frame, text="Adjust Weights (Train)", width=25,
                                        command=self.adjust_weights_button_callback)
        self.reset_weight_button.grid(row=0, column=4)
        self.adjust_weights_button.grid(row=1, column=4)

    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())

    def lambda_slider_callback(self):
        self.lambda_reg = np.int(self.lambda_slider.get())

    def F1_slider_callback(self):
        self.F1 = np.int(self.F1_slider.get())

    def K1_slider_callback(self):
        self.K1 = np.int(self.K1_slider.get())

    def F2_slider_callback(self):
        self.F2 = np.int(self.F2_slider.get())

    def K2_slider_callback(self):
        self.K2 = np.int(self.K2_slider.get())

    def train_data_percentage_slider_callback(self):
        self.train_data_percentage = np.int(self.train_data_percentage_slider.get())
        self.read_train_data()

    def epoch_slider_callback(self):
        self.epoch = np.int(self.epoch_slider.get())

    def reset_weight_button_callback(self):
        self.create_base_model()
        self.error_rate = []

    def create_base_model(self):
        self.error_rate = []
        self.model = Sequential()

        self.model.add(Conv2D(self.F1, (self.K1, self.K1), strides=1, padding='same',
                              kernel_regularizer=regularizers.l2(self.lambda_reg),
                              input_shape=self.train_images.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(self.F2, (self.K2, self.K2), strides=1, padding='same',
                              kernel_regularizer=regularizers.l2(self.lambda_reg)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(
            Conv2D(32, (3, 3), strides=1, padding='same', kernel_regularizer=regularizers.l2(self.lambda_reg),
                   activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(10, kernel_regularizer=regularizers.l2(self.lambda_reg)))
        self.model.add(Activation("softmax"))

        sgd = SGD(lr=self.alpha, decay=1e-6, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.weights = self.model.get_weights()

    def adjust_weights_button_callback(self):
        self.model.set_weights(self.weights)
        cnn = self.model.fit(self.train_images, self.train_labels, validation_data=(self.test_images, self.test_labels),
                             batch_size=self.batch_size, epochs=self.epoch, shuffle=True)
        self.weights = self.model.get_weights()
        for val in cnn.history['val_acc']:
            self.error_rate.append(100 - val*100)
        self.plot_error()
        self.print_confusion_matrix()

    def plot_error(self):
        self.axes.cla()
        self.axes.set_xlabel('epoch')
        self.axes.set_ylabel('error_rate (in %)')
        no_of_epochs = len(self.error_rate)
        print("Error rate: ", self.error_rate)
        plt.xlim(0, no_of_epochs)
        plt.ylim(0, 100)
        if no_of_epochs==1:
            self.axes.plot(1, self.error_rate[0], 'bo', markersize=4)
        self.axes.plot(np.linspace(1, no_of_epochs, no_of_epochs), self.error_rate, 'b-', markersize=1)
        self.axes.xaxis.set_visible(True)
        self.axes.set_title("Error-Rate Graph")
        self.canvas.draw()

    def print_confusion_matrix(self):
        pred = self.model.predict(self.test_images)
        y_pred = []
        for i in pred:
            y_pred.append(np.argmax(i))
        self.display_numpy_array_as_table(confusion_matrix(self.test_label, y_pred))

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

    def read_train_data(self):
        count = np.int((self.train_data_count * self.train_data_percentage) / 100)
        self.train_images = np.empty(shape = [0,3,32,32])
        self.train_labels = []
        for file in os.listdir(self.data_folder):
            if count>0 and fnmatch.fnmatch(file, 'data_*'):
                data = self.unpickle(os.path.join(self.data_folder, file))
                labels = data[b'labels'][0:count]
                self.train_images = np.append(self.train_images, data[b'data'])
                self.train_labels += labels
                count -= 10000
        count = np.int((self.train_data_count * self.train_data_percentage) / 100)
        self.train_images = self.train_images.reshape((math.ceil(count/10000)*10000, 3, 32, 32)).transpose(0, 2, 3, 1)[:count,:,:,:]
        self.train_images = self.train_images.astype('float32')
        self.train_images /= 255
        self.train_labels = np.array(self.train_labels).reshape((len(self.train_labels), 1))
        self.train_labels = K.utils.to_categorical(self.train_labels, 10)
        # print("Train image:",self.train_images.shape)
        # print("Train Label:",self.train_labels.shape)

    def read_test_data(self):
        data = self.unpickle('./Data/test_batch')
        self.test_images = data[b'data'].reshape((10000, 32, 32, 3))
        self.test_label = data[b'labels']
        self.test_labels = K.utils.to_categorical(data[b'labels'], 10)
        self.test_images = self.test_images.astype('float32')
        self.test_images /= 255
        # print("Test image:", self.test_images.shape)
        # print("Test Label:", self.test_labels.shape)

    def unpickle(self, file, encoding='bytes'):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding=encoding)
        return dict

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.title('Assignment_06 --  Singh')
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()