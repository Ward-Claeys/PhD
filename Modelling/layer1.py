#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
layers model
the point was to see if one can do image classification 
when building on a different task output (eg, linear map output, or
living/nonliving output).
the answer thus far is a clear no :)
labels for the image data are
["airplane", "automobile", "bird", "cat",
"deer", "dog", "frog", "horse", "ship", "truck"]
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tom/Documents/Modcogproc/modeling-master/code_by_chapter/Chapter_5')
from ch5_tf2_digit_classif import test_performance, preprocess_digits #, plot_digits,  
from ch5_tf2_image_classif import preprocess_imgs

dataset = "digits"

# import dataset
if dataset == "digits":
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	n_labels   = int(np.max(y_train)+1)
	living     = {2, 3, 4, 5, 6, 7}
	nonliving  = set(range(n_labels)) - living
	randomset  = {0, 1, 2, 3, 4}
	complset   = set(range(n_labels)) - randomset
	set1, set2 = randomset, complset
	set1, set2 = list(set1), list(set2)

# downscale to make data set smaller (and training faster)
train_size, test_size = 1000, 900
x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]

# hyperparameters of the estimation process
learning_rate = 0.002
epochs = 1000
batch_size = 20
batches = int(x_train.shape[0] / batch_size)
activation_function = "softmax"

#plot_digits(x_train[:4], y_train[:4])

if dataset == "digits":
	# pre-processing digits
	n_labels = int(np.max(y_train)+1)
	image_size = x_train.shape[1]*x_train.shape[2] # 798
	x_train, y_train, x_test, y_test = preprocess_digits(
		                                  x_train, y_train, train_size, x_test, y_test, test_size, image_size = image_size, n_labels = n_labels)
else:
	# pre-processing images
	y_train1   = np.isin(y_train, set1).astype(np.int32)
	y_test1    = np.isin(y_test, set2).astype(np.int32)
	n_labels1  = int(np.max(y_train1)+1)
	image_size = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
	x_train, y_train, x_test, y_test = preprocess_imgs(
		                                  x_train, y_train, train_size, x_test, y_test, test_size, image_size = image_size, n_labels = n_labels)
	_, y_train1, _, y_test1 = preprocess_imgs(
		                                  x_train, y_train1, train_size, x_test, y_test1, test_size, image_size = image_size, n_labels = n_labels)


# model construction
def build_layer_model(input_size, n_labels, learning_rate, activation_function):
	model = tf.keras.Sequential([
			tf.keras.Input(shape=(input_size,)),
			tf.keras.layers.Dense(n_labels, activation = activation_function)
			] )
	model.build()
	loss_CE = tf.keras.losses.CategoricalCrossentropy() # don't worry about the details; it's another error function for multi-category, binary data
	opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
	model.compile(optimizer = opt, loss = loss_CE)
	return model


# model fitting
model1   = build_layer_model(image_size, n_labels, learning_rate, activation_function)
history1 = model1.fit(x_train, y_train1, batch_size = batch_size, epochs = epochs)

y_pred      = model1(x_train).numpy() # extra input for model 2
y_test_pred = model1(x_test).numpy()  # same
x_train_extended = np.column_stack((x_train, y_pred)) 
x_test_extended  = np.column_stack((x_test, y_test_pred))
model2   = build_layer_model(image_size + n_labels, n_labels, learning_rate, activation_function) 
history2 = model2.fit(x_train_extended, y_train, batch_size = batch_size, epochs = epochs)

# show output
# print a summary
#model1.summary()
#model2.summary()

# error curve
fig, ax = plt.subplots(1, figsize=(7,3))
ax.plot(history1.history["loss"], color = "red")
ax.plot(history2.history["loss"], color = "black")
test_performance(model1, x_train, x_test, y_train1, y_test1)
test_performance(model2, x_train_extended, x_test_extended, y_train, y_test)

