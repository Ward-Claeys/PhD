#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:50:57 2026

@author: wardclaeys
"""

#Test_test_test

#%%Import packages
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from itertools import product
import pandas as pd
import sys
import time 

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
    
    return loss

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
    opt = keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt

def run_model_1(simulation , model_1_index , opt , index_trial): 
    
    current_trial = input_matrix[index_trial]
    current_output = output_matrix[index_trial]
    
    # take a step in weight space
    loss = step(model_1, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_1[simulation , model_1_index] , accuracy_1[simulation , model_1_index]) = model_1.evaluate(input_matrix , output_matrix , verbose = 0)
            
    ##Get the predictions of the model (for model 2, the sequential one)
    predictions_model = model_1(input_matrix)
    
    return predictions_model , loss 

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
    opt = keras.optimizers.Adam(learning_rate = 0.01)
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
    loss = step(model_2, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_2[simulation , model_2_index] , accuracy_2[simulation , model_2_index]) = model_2.evaluate(input_matrix_2 , output_matrix_2 , verbose = 0)
    
    model_2_predictions = model_2(input_matrix_2)
    
    return model_2_predictions , loss 

#%%Model 3 

history_3 = np.zeros((n_simulations , n_choices))
history_3.fill(-9999)
accuracy_3 = np.zeros((n_simulations , n_choices))
accuracy_3.fill(-9999)

def build_model_3(): 
    
    model = keras.Sequential([
        Input(shape = (10 ,)) , 
        Dense(8 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    return model , opt 

def run_model_3(simulation , model_3_index , opt, index_trial): 
    
    input_matrix_3 = np.column_stack([input_matrix , model_2_prediction])
    
    current_trial = input_matrix_3[index_trial]
    current_output = output_matrix_3[index_trial]
    
    # take a step in weight space
    loss = step(model_3, current_trial, current_output , opt)
    
    # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
    (history_3[simulation , model_3_index] , accuracy_3[simulation , model_3_index]) = model_3.evaluate(input_matrix_3 , output_matrix_3 , verbose = 0)
    
    model_3_predictions = model_3(input_matrix_3)
    
    return model_3_predictions , loss 

#%% Select random trial index from an epoch
def random_trial_index(): 
    index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
    
    return index_trial

#%%Actual model training

#PEt = PE(t - 1) + alpha * (Pt - Pt-1)

#Get the current alpha for smoothing of the PEs 
#alpha_1 = sys.argv[1]

#alpha = (float(alpha_1[0]))

alpha = 0.2

softmax_choices = np.zeros((n_choices , 3 * n_simulations))

model_choices = [1 , 2 , 3]

simulation = 0

data = pd.DataFrame()
choice_order = pd.DataFrame()

weight_PE       = -0.5
weight_LP       = 1
weight_ULP      = 1
weight_novelty  = 1

for simulation in range(n_simulations): 
    
    model_1 , opt_1   = build_model_1()
    model_2 , opt_2   = build_model_2()
    model_3 , opt_3   = build_model_3()
    
    model_1_index = 0
    model_2_index = 0
    model_3_index = 0
    
    choice = 0
    
    novelty_1 = 0
    novelty_2 = 0
    novelty_3 = 0
    
    time_0 = time.clock_gettime(time.CLOCK_REALTIME)
    
    for choice in range(n_choices): 
        
        time_1 = time.clock_gettime(time.CLOCK_REALTIME)
        
        data.loc[choice + n_choices * simulation , "simulation"] = simulation
        
        if choice == 0: 
            #choice_order.loc[ : , ["simulation_1" , "simulation_2" , "simulation_3" , "simulation_4" , "simulation_5" , "simulation_6" , "simulation_7"]] = [0 , 0 , 0 , 0 , 0 , 0 , 0]

            current_model = np.random.choice(model_choices) 
            data.loc[choice + n_choices * simulation , ["PE_1" , "PE_2" , "PE_3"]]                              = [0 , 0 , 0]
            #data.loc[choice + n_choices * simulation , ["smoothed_PE_1" , "smoothed_PE_2" , "smoothed_PE_3"]]   = [0 , 0 , 0]
            data.loc[choice + n_choices * simulation , ["LP_1" , "LP_2" , "LP_3"]]                              = [0 , 0 , 0]
            data.loc[choice + n_choices * simulation , ["novelty_1" , "novelty_2" , "novelty_3"]]               = [0 , 0 , 0]
            data.loc[choice + n_choices * simulation , ["Model_1" , "Model_2" , "Model_3"]]                     = [0 , 0 , 0]
            
            #For later use to not get an error, since I don't really calculate this now. 
            #I lock all the smoothed errors in the actual model training, so there I just always want to lock the error in the file
            error_1 , error_2 , error_3  = [0 , 0 , 0]
            
            softmax_choices[choice , 0 + 3 * simulation : 3 + 3 * simulation] = [0 , 0 , 0]
            
        else: 
            if choice == 1: 
                data.loc[choice + n_choices * simulation , ["LP_1" , "LP_2" , "LP_3"]] = [0 , 0 , 0]
                
                error_1 = data.loc[choice + n_choices * simulation - 1 , "PE_1"]
                error_2 = data.loc[choice + n_choices * simulation - 1 , "PE_2"]
                error_3 = data.loc[choice + n_choices * simulation - 1 , "PE_3"]
                
                #In the previous loop, I locked zero as the PE, but want to get the actual PE in there, so say that on the previous trial, this was the smoothed PE
                #data.loc[choice + n_choices * simulation - 1 , ["smoothed_PE_1" , "smoothed_PE_2" , "smoothed_PE_3"]] = [error_1 , error_2 , error_3]
                
                error_list = [error_1 , error_2 , error_3]
                error_list = [i * weight_PE for i in error_list]
                
                #Transfer them to softmax values 
                model_options = np.exp(error_list)/np.sum(np.exp(error_list))
                
                #I do plus one because the np.argmax thingie does a zero-based indexing whilst it's model 1, 2 and 3
                current_model = np.random.choice(3, p = model_options) + 1 
                
                data.loc[choice + n_choices * simulation , ["Model_1" , "Model_2" , "Model_3"]] = model_options
                
                softmax_choices[choice , 0 + 3 * simulation : 3 + 3 * simulation] = model_options
                
            else: 
                
                #From here on out I start smoothing the PEs, couldn't earlier because there were no previous PEs
                #Alpha now 0.5, see what happens and how to adjust 
                """
                error_1 = data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_1"] + alpha * (data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_1"] - data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_1"])
                error_2 = data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_2"] + alpha * (data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_2"] - data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_2"])
                error_3 = data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_3"] + alpha * (data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_3"] - data.loc[choice + n_choices * simulation - 2 , "smoothed_PE_3"])
                """
                error_1 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_1"]
                error_2 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_2"]
                error_3 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_3"]
                
                #data.loc[choice + n_choices * simulation  - 1  , ["smoothed_PE_1" , "smoothed_PE_2" , "smoothed_PE_3"]] = [error_1 , error_2 , error_3]
                
                """
                error_1 = data.loc[choice + n_choices * simulation - 1 , "PE_1"]
                error_2 = data.loc[choice + n_choices * simulation - 1 , "PE_2"]
                error_3 = data.loc[choice + n_choices * simulation - 1 , "PE_3"]
                """
                
                error_list = [error_1 , error_2 , error_3]
                error_list = [i * weight_PE for i in error_list]
                
                LP_1 = data.loc[choice + n_choices * simulation - 1 , "LP_1"]
                LP_2 = data.loc[choice + n_choices * simulation - 1 , "LP_2"]
                LP_3 = data.loc[choice + n_choices * simulation - 1 , "LP_3"]
                
                #data.loc[choice + n_choices * simulation , ["LP_1" , "LP_2" , "LP_3"]] = [LP_1 , LP_2 , LP_3]
                
                lp_list = [LP_1 , LP_2 , LP_3]
                lp_list = [i * weight_LP for i in lp_list]
                
                ULP_1 = np.abs(LP_1)
                ULP_2 = np.abs(LP_2)
                ULP_3 = np.abs(LP_3)
                
                data.loc[choice + n_choices * simulation , ["ULP_1" , "ULP_2" , "ULP_3"]] = [ULP_1 , ULP_2 , ULP_3]
                
                novelty_1_exp = np.exp(-novelty_1)
                novelty_2_exp = np.exp(-novelty_2)
                novelty_3_exp = np.exp(-novelty_3)
                
                numerator_model_1 = np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_novelty * novelty_1_exp)
                numerator_model_2 = np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_novelty * novelty_2_exp)
                numerator_model_3 = np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_novelty * novelty_3_exp)
                
                model_1_value = numerator_model_1 / np.sum([numerator_model_1 , numerator_model_2 , numerator_model_3])
                model_2_value = numerator_model_2 / np.sum([numerator_model_1 , numerator_model_2 , numerator_model_3])
                model_3_value = numerator_model_3 / np.sum([numerator_model_1 , numerator_model_2 , numerator_model_3])
                
                
                #model_1_value = np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_novelty * novelty_1_exp) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_novelty * novelty_3_exp))
                #model_2_value = np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_novelty * novelty_2_exp) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_novelty * novelty_3_exp))
                #model_3_value = np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_novelty * novelty_3_exp) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_novelty * novelty_3_exp))
                
                """
                model_1_value = np.exp(weight_PE * error_1 + weight_novelty * novelty_1_exp) / (np.exp(weight_PE * error_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_novelty * novelty_3_exp))
                model_2_value = np.exp(weight_PE * error_2 + weight_novelty * novelty_2_exp) / (np.exp(weight_PE * error_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_novelty * novelty_3_exp))
                model_3_value = np.exp(weight_PE * error_3 + weight_novelty * novelty_3_exp) / (np.exp(weight_PE * error_1 + weight_novelty * novelty_1_exp) + np.exp(weight_PE * error_2 + weight_novelty * novelty_2_exp) + np.exp(weight_PE * error_3 + weight_novelty * novelty_3_exp))
                """
                
                #Transfer them to softmax values 
                model_options = [model_1_value , model_2_value , model_3_value]
                
                data.loc[choice + n_choices * simulation , ["Model_1" , "Model_2" , "Model_3"]] = model_options
                
                #I do plus one because the np.argmax thingie does a zero-based indexing whilst it's model 1, 2 and 3
                current_model = np.random.choice(3, p = model_options) + 1 
                
                softmax_choices[choice , 0 + 3 * simulation : 3 + 3 * simulation] = model_options
        
        time_2 = time.clock_gettime(time.CLOCK_REALTIME)
        
        #print(time_2 - time_1)
        
        data.loc[choice + n_choices * simulation , "model_choice"] = current_model
        
        string = "simulation_" + str(simulation)
        
        choice_order.loc[choice , string] = current_model
        
        #get random index to select the trial
        index_trial = random_trial_index()
        
        if current_model == 1: 
            
            #Update novelty values
            novelty_1 += 1
            novelty_2 , novelty_3 = 0 , 0
            
            #Model with previous predictions
            model_prediction , loss_1 = run_model_1(simulation , model_1_index , opt_1 , index_trial)
            
            data.loc[choice + n_choices * simulation , "PE_1"] = loss_1.numpy()
            
            if choice != 0: 
                data.loc[choice + n_choices * simulation , "PE_2"] = data.loc[choice + n_choices * simulation - 1 , "PE_2"]
                data.loc[choice + n_choices * simulation , "PE_3"] = data.loc[choice + n_choices * simulation - 1 , "PE_3"]
                
                data.loc[choice + n_choices * simulation , ["novelty_1" , "novelty_2" , "novelty_3"]] = [novelty_1 , novelty_2 , novelty_3]
            
            if choice > 1: 
                data.loc[choice + n_choices * simulation , "LP_1"] =  data.loc[choice + n_choices * simulation - 1 , "PE_1"] - data.loc[choice + n_choices * simulation , "PE_1"]
                
                #Copy down the LP's of the unchosen models
                data.loc[choice + n_choices * simulation , "LP_2"] = data.loc[choice + n_choices * simulation - 1 , "LP_2"]
                data.loc[choice + n_choices * simulation , "LP_3"] = data.loc[choice + n_choices * simulation - 1 , "LP_3"]
            
            model_1_index += 1
            
        elif current_model == 2: 
            
            if model_1_index == 0: 
                model_prediction = np.zeros((64 , 2))
            
            #Update novelty values
            novelty_2 += 1
            novelty_1 , novelty_3 = 0 , 0
            
            model_2_prediction , loss_2 = run_model_2(simulation , model_2_index , opt_2 , index_trial)
            
            data.loc[choice + n_choices * simulation , "PE_2"] = loss_2.numpy()
            
            if choice != 0: 
                data.loc[choice + n_choices * simulation , "PE_1"] = data.loc[choice + n_choices * simulation - 1 , "PE_1"]
                data.loc[choice + n_choices * simulation , "PE_3"] = data.loc[choice + n_choices * simulation - 1 , "PE_3"]
                
                data.loc[choice + n_choices * simulation , ["novelty_1" , "novelty_2" , "novelty_3"]] = [novelty_1 , novelty_2 , novelty_3]

            if choice > 1: 
                data.loc[choice + n_choices * simulation , "LP_2"] =  data.loc[choice + n_choices * simulation - 1 , "PE_2"] - data.loc[choice + n_choices * simulation , "PE_2"]
                
                #Copy down the LP's of the unchosen models
                data.loc[choice + n_choices * simulation , "LP_1"] = data.loc[choice + n_choices * simulation - 1 , "LP_1"]
                data.loc[choice + n_choices * simulation , "LP_3"] = data.loc[choice + n_choices * simulation - 1 , "LP_3"]
            
            model_2_index += 1
            
        elif current_model == 3: 
            
            if model_2_index == 0: 
                model_2_prediction = np.zeros((64 , 4))
            
            #Update novelty values
            novelty_3 += 1
            novelty_1 , novelty_2 = 0 , 0
            
            model_3_prediction , loss_3 = run_model_3(simulation , model_3_index , opt_3 , index_trial)
            
            data.loc[choice + n_choices * simulation , "PE_3"] = loss_3.numpy()
            
            if choice != 0: 
                data.loc[choice + n_choices * simulation , "PE_1"] = data.loc[choice + n_choices * simulation - 1 , "PE_1"]
                data.loc[choice + n_choices * simulation , "PE_2"] = data.loc[choice + n_choices * simulation - 1 , "PE_2"]
                
                data.loc[choice + n_choices * simulation , ["novelty_1" , "novelty_2" , "novelty_3"]] = [novelty_1 , novelty_2 , novelty_3]
            
            if choice > 1: 
                data.loc[choice + n_choices * simulation , "LP_3"] =  data.loc[choice + n_choices * simulation - 1 , "PE_3"] - data.loc[choice + n_choices * simulation , "PE_3"]
                
                #Copy down the LP's of the unchosen models
                data.loc[choice + n_choices * simulation , "LP_1"] = data.loc[choice + n_choices * simulation - 1 , "LP_1"]
                data.loc[choice + n_choices * simulation , "LP_2"] = data.loc[choice + n_choices * simulation - 1 , "LP_2"]
            
            model_3_index += 1 
        
        time_3 = time.clock_gettime(time.CLOCK_REALTIME)
        
        #print(time_3 - time_2)
        
        #data.loc[choice + n_choices * simulation , ["smoothed_PE_1" , "smoothed_PE_2" , "smoothed_PE_3"]] = [error_1 , error_2 , error_3]
    
        if choice != 0: 
            #Make sure the last row of the smoothed PEs get filled too, so quick fix here 
            error_1 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_1"] + alpha * (data.loc[choice + n_choices * simulation , "PE_1"] - data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_1"])
            error_2 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_2"] + alpha * (data.loc[choice + n_choices * simulation , "PE_2"] - data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_2"])
            error_3 = data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_3"] + alpha * (data.loc[choice + n_choices * simulation , "PE_3"] - data.loc[choice + n_choices * simulation - 1 , "smoothed_PE_3"])
            
        else: 
            
            error_1 = data.loc[choice + n_choices * simulation , "PE_1"] 
            error_2 = data.loc[choice + n_choices * simulation , "PE_2"]
            error_3 = data.loc[choice + n_choices * simulation , "PE_3"]
            
        data.loc[choice + n_choices * simulation , ["smoothed_PE_1" , "smoothed_PE_2" , "smoothed_PE_3"]] = [error_1 , error_2 , error_3]
        
        time_4 = time.clock_gettime(time.CLOCK_REALTIME)
        
        #print(time_4 - time_1)
    
    time_5 = time.clock_gettime(time.CLOCK_REALTIME)
    
    print(time_5 - time_0)
    
    directory_to_write = "data/alpha_" + str(alpha)
    
    data.to_csv(directory_to_write + '/dataframe_testing.csv')
    
    choice_order.to_csv(directory_to_write + '/order_testing.csv')
    
    np.save(directory_to_write + "/model_loss_1" , history_1)
    np.save(directory_to_write + "/model_loss_2" , history_2)
    np.save(directory_to_write + "/model_loss_3" , history_3)
    
    np.save(directory_to_write + "/model_accuracy_1" , accuracy_1)
    np.save(directory_to_write  + "/model_accuracy_2" , accuracy_2)
    np.save(directory_to_write + "/model_accuracy_3" , accuracy_3)
    
    np.save(directory_to_write + "/softmax_choices" , softmax_choices)

"""
#part of code when you still need to change your working directory. 

import os
os.chdir("/Users/wardclaeys/OneDrive - UGent/Modelling")
"""
