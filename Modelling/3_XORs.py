#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 09:10:59 2025

@author: wardclaeys
"""

#%%
import numpy as np

#Oké, itertools is wel echt zeer handig 😃
from itertools import product

input_options = np.asarray(list(product([0, 1], repeat = 6)))

##Construct input matrix, will be used for all models
input_matrix = np.zeros((64 , 6))

input_matrix[ : , 0 : 6] = input_options

##Construct output matrix for model 1 (I know not efficient, but just wanted to check if it worked). 
output_matrix = np.zeros((64 , 2))

for i in range(output_matrix.shape[0]): 
    if ((input_matrix[i , 0] == 0) and (input_matrix[i , 1] == 1)) or ((input_matrix[i , 0] == 1) and (input_matrix[i , 1] == 0)): 
        output_matrix[i , 0] = 1

#If left is 1, right is 0 and reversed
output_matrix[ : , 1] = 1 - output_matrix[: , 0]

#%%Basics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from itertools import product

n_epochs = 150
n_shuffles = 100

#%% Model 1
history_1 = np.zeros((n_shuffles , n_epochs))

#Function for building model 1 
def build_model(): 
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(2 , activation = "softmax")
         ])
    model.build()
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, loss = loss_CE)
    
    return model


for shuffle in range(n_shuffles): 
    #Build it
    model = build_model()
    
    #Train for 70 epochs
    history1 = model.fit(input_matrix, output_matrix, batch_size = 1, epochs = n_epochs)
    #plt.plot(history1.history["loss"])
    
    history_1[shuffle , : ] = history1.history["loss"]
    
##Get the predictions of the model (for model 2, the sequential one)
predictions_model = model.predict(input_matrix)

mean_1, lower_1, upper_1 = [],[],[]

ci = 0.95

for i in range (history_1.shape[1]):

    a = history_1[ : , i]

    MEAN = np.mean(a)
    mean_1.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_1.append(Lower)
    upper_1.append(Upper)

"""
plt.plot(mean_1 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_1))), upper_1, lower_1, color="green", alpha=0.25)
plt.show()
"""
#%%Model 2

history_2_itself_matrix = np.zeros((n_shuffles , n_epochs))

##Construct the output matrix for model 2 (4 outputs). Again not efficient, but was again just to check it worked. 
output_matrix_2 = np.zeros((64 , 4))

for i in range(output_matrix_2.shape[0]): 
    if (input_matrix[i , 0] == 0 and input_matrix[i , 1] == 1) or (input_matrix[i , 0] == 1 and input_matrix[i , 1] == 0):
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            output_matrix_2[i , : ] = [1 , 0 , 0 , 0]
        else: 
            output_matrix_2[i , : ] = [0, 1 , 0 , 0]
    else: 
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            output_matrix_2[i , : ] = [0 , 0 , 1 , 0]
        else: 
            output_matrix_2[i , : ] = [0 , 0 , 0 , 1]

#Build model 2 that trains by itself, so 5 inputs to 4 outputs with a hidden layer in between (not sure if this should be added, but otherwise the XOR is impossible, so just left it in for now)
def model_2_itself(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, loss = loss_CE)
    
    return model

for shuffle in range(n_shuffles): 
    #Build the model
    model_2 = model_2_itself()
    
    #And train it 
    history_2_itself = model_2.fit(input_matrix , output_matrix_2 , batch_size = 1 , epochs = n_epochs)
    #plt.plot(history_2_itself.history["loss"])
    
    history_2_itself_matrix[shuffle , : ] = history_2_itself.history["loss"]

mean_2, lower_2, upper_2 = [],[],[]

ci = 0.95

for i in range (history_2_itself_matrix.shape[1]):

    a = history_2_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_2.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_2.append(Lower)
    upper_2.append(Upper)
"""
plt.plot(mean_2 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_2))), upper_2, lower_2, color="green", alpha=0.25)
plt.show()
"""
#%% Model 2 sequential

history_2_sequential_matrix = np.zeros((n_shuffles , n_epochs))

#Then build model 2 for sequential learning. Here the input matrix is different with an extra 2 units, based on model 1
#Note that here the hidden layer is gone, and it's just the 7 (5 + 2) inputs on 4 outputs. Hidden layer is so to say hidden-hidden in the 2 new input units
def model_2_sequential(): 
    
    model = keras.Sequential([
        Input(shape = (8 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, loss = loss_CE)
    
    return model

#Make the new input unit with the predictions of the model 
input_matrix_2 = np.column_stack([input_matrix , predictions_model])

for shuffle in range(n_shuffles): 
    #Build the model 
    model_2 = model_2_sequential()
    
    #Train it
    history_2_sequential = model_2.fit(input_matrix_2 , output_matrix_2 , batch_size = 1 , epochs = n_epochs)
    #plt.plot(history_2_sequential.history["loss"])
    history_2_sequential_matrix[shuffle , : ] = history_2_sequential.history["loss"]

model_2_predictions = model_2.predict(input_matrix_2)

mean_3, lower_3, upper_3 = [],[],[]

ci = 0.95

for i in range (history_2_sequential_matrix.shape[1]):

    a = history_2_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_3.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_3.append(Lower)
    upper_3.append(Upper)

"""
plt.plot(mean_3 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_3))), upper_3, lower_3, color="green", alpha=0.25)
plt.show()
"""
#%%Model 3 sequential


##########################
## Make output matrix 3 ##
##########################

output_matrix_3 = np.zeros((64 , 8))

for i in range(output_matrix_3.shape[0]): 
    if (input_matrix[i , 0] == 0 and input_matrix[i , 1] == 1) or (input_matrix[i , 0] == 1 and input_matrix[i , 1] == 0):
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0): 
                output_matrix_3[i , : ] = [1 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0]
        else: 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0):
                output_matrix_3[i , : ] = [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0]

    else: 
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0): 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]
        else: 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0):
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1]

history_3_sequential_matrix = np.zeros((n_shuffles , n_epochs))

def model_3_sequential(): 
    
    model = keras.Sequential([
        Input(shape = (10 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, loss = loss_CE)
    
    return model

input_matrix_3 = np.column_stack([input_matrix , model_2_predictions])


for shuffle in range(n_shuffles): 
    #Build the model 
    model_3 = model_3_sequential()
    
    #Train it
    history_3_sequential = model_3.fit(input_matrix_3 , output_matrix_3 , batch_size = 1 , epochs = n_epochs)
    #plt.plot(history_2_sequential.history["loss"])
    history_3_sequential_matrix[shuffle , : ] = history_3_sequential.history["loss"]


mean_4, lower_4, upper_4 = [],[],[]

ci = 0.95

for i in range (history_3_sequential_matrix.shape[1]):

    a = history_3_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_4.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_4.append(Lower)
    upper_4.append(Upper)
"""
plt.plot(mean_4 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_4))), upper_4, lower_4, color="green", alpha=0.25)
plt.show()
"""
#%% Model 3 by itself

history_3_itself_matrix = np.zeros((n_shuffles , n_epochs))

#Build model 2 that trains by itself, so 5 inputs to 4 outputs with a hidden layer in between (not sure if this should be added, but otherwise the XOR is impossible, so just left it in for now)
def model_3_itself(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, loss = loss_CE)
    
    return model

for shuffle in range(n_shuffles): 
    #Build the model
    model_3 = model_3_itself()
    
    #And train it 
    history_3_itself = model_3.fit(input_matrix , output_matrix_3 , batch_size = 1 , epochs = n_epochs)
    #plt.plot(history_2_itself.history["loss"])
    
    history_3_itself_matrix[shuffle , : ] = history_3_itself.history["loss"]

mean_5, lower_5, upper_5 = [],[],[]

ci = 0.95

for i in range (history_3_itself_matrix.shape[1]):

    a = history_3_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_5.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_5.append(Lower)
    upper_5.append(Upper)

"""
plt.plot(mean_5 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_5))), upper_5, lower_5, color="green", alpha=0.25)
plt.show()
"""
#%%

fig , ax = plt.subplots(1 , 3)

ax[0].plot(mean_1 , "-b" , label = "mean" , color = "green")
ax[0].fill_between(list(range(len(mean_1))), upper_1, lower_1, color="green", alpha=0.25)
ax[0].set_title("Model 1")

ax[1].plot(mean_2 , "-b" , label = "mean" , color = "red")
ax[1].fill_between(list(range(len(mean_2))), upper_2, lower_2, color="red", alpha=0.25)
ax[1].plot(mean_3 , "-b" , label = "mean" , color = "green")
ax[1].fill_between(list(range(len(mean_3))), upper_3, lower_3, color="green", alpha=0.25)
ax[1].set_title("Model 2")

ax[2].plot(mean_4 , "-b" , label = "mean" , color = "green")
ax[2].fill_between(list(range(len(mean_4))), upper_4, lower_4, color="green", alpha=0.25)
ax[2].plot(mean_5 , "-b" , label = "mean" , color = "red")
ax[2].fill_between(list(range(len(mean_5))), upper_5, lower_5, color="red", alpha=0.25)
ax[2].set_title("Model 3")

plt.suptitle("Model losses (red without training earlier models, green with)")

