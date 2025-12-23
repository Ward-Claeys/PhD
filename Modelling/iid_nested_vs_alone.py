#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:57:56 2025

@author: wardclaeys
"""

#%%Import packages
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


#%% Make input matrix (used in all models)
input_options = np.asarray(list(product([0, 1], repeat = 6)))

##Construct input matrix, will be used for all models
input_matrix = np.zeros((64 , 6))

input_matrix[ : , 0 : 6] = input_options


#%%Construct all the output matrices. Different difficulties represented with more complex rules and more output units. 

##Construct output matrix for model 1 (I know not efficient, but just wanted to check if it worked). 
output_matrix = np.zeros((64 , 2))

for i in range(output_matrix.shape[0]): 
    if ((input_matrix[i , 0] == 0) and (input_matrix[i , 1] == 1)) or ((input_matrix[i , 0] == 1) and (input_matrix[i , 1] == 0)): 
        output_matrix[i , 0] = 1

#If left is 1, right is 0 and reversed
output_matrix[ : , 1] = 1 - output_matrix[: , 0]

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

##Construct the output matrix for model 3 (8 outputs). Again not efficient, but was again just to check it worked. 
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

#%% Define the step we take in weight space
def step(model, X, y , opt):
    """keep track of our gradients"""
    with tf.GradientTape() as tape:
     # make a prediction using the model and then calculate the loss
#        dict_X = array_to_dict(X)
        pred = model(inputs = X) 
        loss = categorical_crossentropy(y, pred)
	    # calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

#%%Basics

n_choices   = 10000
n_simulations  = 10

#%% Model 1
history_1 = np.zeros((n_simulations , n_choices))
history_1.fill(-9999)
accuracy_1 = np.zeros((n_simulations , n_choices))
accuracy_1.fill(-9999)

#Function for building model 1 
def build_model_1(): 
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(2 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt

def run_model_1(simulation , model_1_index , opt , index_trial): 
    
    current_trial = input_matrix[index_trial]
    current_output = output_matrix[index_trial]
    
    # take a step in weight space
    step(model_1, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_1[simulation , model_1_index] , accuracy_1[simulation , model_1_index]) = model_1.evaluate(input_matrix , output_matrix)
            
    ##Get the predictions of the model (for model 2, the sequential one)
    predictions_model = model_1(input_matrix)
    
    return predictions_model

#%% Model 2 

history_2 = np.zeros((n_simulations , n_choices))
history_2.fill(-9999)
accuracy_2= np.zeros((n_simulations , n_choices))
accuracy_2.fill(-9999)

#Then build model 2 for sequential learning. Here the input matrix is different with an extra 2 units, based on model 1
#Note that here the hidden layer is gone, and it's just the 7 (5 + 2) inputs on 4 outputs. Hidden layer is so to say hidden-hidden in the 2 new input units
def build_model_2(): 
    
    model = keras.Sequential([
        Input(shape = (8 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt 

def run_model_2(simulation , model_2_index , opt , index_trial): 
    #Make the new input unit with the predictions of the model 
    input_matrix_2 = np.column_stack([input_matrix , model_prediction])
    
    current_trial = input_matrix_2[index_trial]
    current_output = output_matrix_2[index_trial]
    
    # take a step in weight space
    step(model_2, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_2[simulation , model_2_index] , accuracy_2[simulation , model_2_index]) = model_2.evaluate(input_matrix_2 , output_matrix_2)
    
    model_2_predictions = model_2(input_matrix_2)
    
    return model_2_predictions

#%%Model 3 

history_3 = np.zeros((n_simulations , n_choices))
history_3.fill(-9999)
accuracy_3 = np.zeros((n_simulations , n_choices))
accuracy_3.fill(-9999)

def build_model_3(): 
    
    model = keras.Sequential([
        Input(shape = (10 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt 

def run_model_3(simulation , model_3_index , opt, index_trial): 
    
    input_matrix_3 = np.column_stack([input_matrix , model_2_prediction])
    
    current_trial = input_matrix_3[index_trial]
    current_output = output_matrix_3[index_trial]
    
    # take a step in weight space
    step(model_3, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_3[simulation , model_3_index] , accuracy_3[simulation , model_3_index]) = model_3.evaluate(input_matrix_3 , output_matrix_3)
    
    model_3_predictions = model_3(input_matrix_3)
    
    return model_3_predictions

#%% Model 2 solo quest

history_2_alone = np.zeros((n_simulations , n_choices))
history_2_alone.fill(-9999)
accuracy_2_alone= np.zeros((n_simulations , n_choices))
accuracy_2_alone.fill(-9999)

#Then build model 2 for sequential learning. Here the input matrix is different with an extra 2 units, based on model 1
#Note that here the hidden layer is gone, and it's just the 7 (5 + 2) inputs on 4 outputs. Hidden layer is so to say hidden-hidden in the 2 new input units
def build_model_2_alone(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt 

def run_model_2_alone(simulation , model_2_index , opt , index_trial): 
    
    index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
    current_trial = input_matrix[index_trial]
    current_output = output_matrix_2[index_trial]
    
    # take a step in weight space
    step(model_2_2, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_2_alone[simulation , model_2_index] , accuracy_2_alone[simulation , model_2_index]) = model_2_2.evaluate(input_matrix , output_matrix_2)
    

#%%Model 3 solo quest

history_3_alone = np.zeros((n_simulations , n_choices))
history_3_alone.fill(-9999)
accuracy_3_alone = np.zeros((n_simulations , n_choices))
accuracy_3_alone.fill(-9999)

def build_model_3_alone(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt 

def run_model_3_alone(simulation , model_3_index , opt , index_trial): 
    
    current_trial = input_matrix[index_trial]
    current_output = output_matrix_3[index_trial]
    
    # take a step in weight space
    step(model_3_2, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_3_alone[simulation , model_3_index] , accuracy_3_alone[simulation , model_3_index]) = model_3_2.evaluate(input_matrix , output_matrix_3)

#%%Not taking the prediction with me training 

def random_trial_index(): 
    index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
    
    return index_trial

#%%Actual model training

model_choices = [1 , 2 , 3]

simulation = 0
choice = 0

for simulation in range(n_simulations): 
    
    model_1 , opt_1   = build_model_1()
    model_2 , opt_2   = build_model_2()
    model_3 , opt_3   = build_model_3()
    
    model_2_2 , opt_2_2   = build_model_2_alone()
    model_3_2 , opt_3_2   = build_model_3_alone()
    
    model_1_index = 0
    model_2_index = 0
    model_3_index = 0
    
    for choice in range(n_choices): 
        
        current_model = np.random.choice( model_choices) 
        #get random index to select the trial
        index_trial = random_trial_index()
        
        if current_model == 1: 
            
            model_1_index += 1
            
            #Model with previous predictions
            model_prediction = run_model_1(simulation , model_1_index , opt_1 , index_trial)
        
        elif current_model == 2: 
            
            if model_1_index == 0: 
                model_prediction = np.zeros((64 , 2))
                
            model_2_index += 1
            
            model_2_prediction = run_model_2(simulation , model_2_index , opt_2 , index_trial)
            
            #Model without them
            run_model_2_alone(simulation , model_2_index , opt_2_2 , index_trial)
        
        elif current_model == 3: 
            
            if model_2_index == 0: 
                model_2_prediction = np.zeros((64 , 4))
            
            model_3_index += 1 
            
            model_3_prediction = run_model_3(simulation , model_3_index , opt_3 , index_trial)
            
            #Model without them
            run_model_3_alone(simulation , model_3_index , opt_3_2 , index_trial)

#%%Prepare for plotting and actual plotting 

mean_1, lower_1, upper_1 = [],[],[]
mean_2, lower_2, upper_2 = [],[],[]

ci = 0.95

for i in range (accuracy_1.shape[1]):

    a = history_1[ : , i][history_1[: , i] != -9999]

    MEAN = np.mean(a)
    mean_1.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_1.append(Lower)
    upper_1.append(Upper)
    
    a = accuracy_1[ : , i][accuracy_1[: , i] != -9999]

    MEAN = np.mean(a)
    mean_2.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_2.append(Lower)
    upper_2.append(Upper)

mean_3, lower_3, upper_3 = [],[],[]
mean_4, lower_4, upper_4 = [],[],[]

mean_7, lower_7, upper_7 = [],[],[]
mean_8, lower_8, upper_8 = [],[],[]

for i in range (accuracy_2.shape[1]):

    a = history_2[ : , i][history_2[: , i] != -9999]

    MEAN = np.mean(a)
    mean_3.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_3.append(Lower)
    upper_3.append(Upper)
    
    a = accuracy_2[ : , i][accuracy_2[: , i] != -9999]

    MEAN = np.mean(a)
    mean_4.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_4.append(Lower)
    upper_4.append(Upper)
    
    a = history_2_alone[ : , i][history_2_alone[: , i] != -9999]

    MEAN = np.mean(a)
    mean_7.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_7.append(Lower)
    upper_7.append(Upper)
    
    a = accuracy_2_alone[ : , i][accuracy_2_alone[: , i] != -9999]

    MEAN = np.mean(a)
    mean_8.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_8.append(Lower)
    upper_8.append(Upper)
    

mean_5, lower_5, upper_5 = [],[],[]
mean_6, lower_6, upper_6 = [],[],[]

mean_9, lower_9, upper_9 = [],[],[]
mean_10, lower_10, upper_10 = [],[],[]

for i in range (accuracy_3.shape[1]):

    a = history_3[ : , i][history_3[: , i] != -9999]

    MEAN = np.mean(a)
    mean_5.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_5.append(Lower)
    upper_5.append(Upper)
    
    a = accuracy_3[ : , i][accuracy_3[: , i] != -9999]

    MEAN = np.mean(a)
    mean_6.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_6.append(Lower)
    upper_6.append(Upper)
    
    a = history_3_alone[ : , i][history_3_alone[: , i] != -9999]

    MEAN = np.mean(a)
    mean_9.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_9.append(Lower)
    upper_9.append(Upper)
    
    a = accuracy_3_alone[ : , i][accuracy_3_alone[: , i] != -9999]

    MEAN = np.mean(a)
    mean_10.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_10.append(Lower)
    upper_10.append(Upper)

fig , ax = plt.subplots(2 , 3)

#Plot the loss functions
ax[0 , 0].plot(mean_1 , "-b" , label = "mean" , color = "red")
ax[0 , 0].fill_between(list(range(len(mean_1))), upper_1, lower_1, color="red", alpha=0.25)
ax[0 , 0].set_title("Model 1")
ax[0 , 0].set_ylabel("Categorical Cross Entropy")
ax[0 , 0].set_ylim(-0.2 , 2.5)

ax[1 , 0].plot(mean_2 , "-b" , label = "mean" , color = "green")
ax[1 , 0].fill_between(list(range(len(mean_2))), upper_2, lower_2, color="green", alpha=0.25)
ax[1 , 0].set_title("Model 1")
ax[1 , 0].set_ylabel("Accuracy")
ax[1 , 0].set_ylim(-0.1 , 1.1)

ax[0 , 1].plot(mean_3 , "-b" , label = "mean" , color = "green")
ax[0 , 1].fill_between(list(range(len(mean_3))), upper_3, lower_3, color="green", alpha=0.25)
ax[0 , 1].plot(mean_7 , "-b" , label = "mean" , color = "red")
ax[0 , 1].fill_between(list(range(len(mean_7))), upper_7, lower_7, color="red", alpha=0.25)
ax[0 , 1].set_title("Model 2")
ax[0 , 1].set_ylabel("Categorical Cross Entropy")
ax[0 , 1].set_ylim(-0.2 , 2.5)

ax[1 , 1].plot(mean_4 , "-b" , label = "mean" , color = "green")
ax[1 , 1].fill_between(list(range(len(mean_4))), upper_4, lower_4, color="green", alpha=0.25)
ax[1 , 1].plot(mean_8 , "-b" , label = "mean" , color = "red")
ax[1 , 1].fill_between(list(range(len(mean_8))), upper_8, lower_8, color="red", alpha=0.25)
ax[1 , 1].set_title("Model 2")
ax[1 , 1].set_ylabel("Accuracy")
ax[1 , 1].set_ylim(-0.1 , 1.1)

ax[0 , 2].plot(mean_5 , "-b" , label = "mean" , color = "green")
ax[0 , 2].fill_between(list(range(len(mean_5))), upper_5, lower_5, color="green", alpha=0.25)
ax[0 , 2].plot(mean_9 , "-b" , label = "mean" , color = "red")
ax[0 , 2].fill_between(list(range(len(mean_9))), upper_9, lower_9, color="red", alpha=0.25)
ax[0 , 2].set_title("Model 3")
ax[0 , 2].set_ylabel("Categorical Cross Entropy")
ax[0 , 2].set_ylim(-0.2 , 2.5)

ax[1 , 2].plot(mean_6 , "-b" , label = "mean" , color = "green")
ax[1 , 2].fill_between(list(range(len(mean_6))), upper_6, lower_6, color="green", alpha=0.25)
ax[1 , 2].plot(mean_10 , "-b" , label = "mean" , color = "red")
ax[1 , 2].fill_between(list(range(len(mean_10))), upper_10, lower_10, color="red", alpha=0.25)
ax[1 , 2].set_title("Model 3")
ax[1 , 2].set_ylabel("Accuracy")
ax[1 , 2].set_ylim(-0.1 , 1.5)

plt.suptitle("Model losses and accuracies (green are models with input of previous models, red without)")

fig.show()

