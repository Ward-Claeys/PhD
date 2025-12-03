#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:34:02 2025

@author: wardclaeys
"""

###########################################################################
## Trying to model without keras by using the formulas of the delta rule ##
###########################################################################

#%% Cell with all definitions

##Import package (singular ;) ) 
import numpy as np

##Softmax function to change the inputs to decisions 
def softmax(x):
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

##Function to get the outputs at the hidden layers 
def model_predict(current_input , model_1_weights): 
    return np.matmul(current_input , model_1_weights)

##Make the full input layers (sometimes a hidden layer and the input units become the input layer "together")
def make_input_layer(x , y): 
    return np.concatenate((x , y))

##Construct the input matrix (used for all models)
def general_initialization(): 
    
    input_matrix = np.zeros((8 , 4))
    
    input_options = [[0 , 0 , 0] , [0 , 0 , 1] , [0 , 1 , 0] , [0 , 1 , 1] , [1 , 0 , 0] , [1 , 0 , 1] , [1 , 1 , 0] , [1 , 1 , 1]]


    ##And fill it
    input_matrix[0 , 0 : 3]    = input_options[0]
    input_matrix[1 , 0 : 3]    = input_options[1]
    input_matrix[2 , 0 : 3]    = input_options[2]
    input_matrix[3 , 0 : 3]    = input_options[3]

    input_matrix[4 , 0 : 3]    = input_options[4]
    input_matrix[5 , 0 : 3]    = input_options[5]
    input_matrix[6 , 0 : 3]    = input_options[6]
    input_matrix[7 , 0 : 3]    = input_options[7]

    input_matrix[ : , 3]    = 1
        
    return input_matrix 

def initialize_all_outputs(): 
    
    output_matrix_1 = np.zeros((8 , 2))

    output_matrix_1[ : , 0]    = 1 - np.transpose(1 * (input_matrix[ : , 0] == 0))
    output_matrix_1[ : , 1]    = 1 - output_matrix_1[ : , 0]
    
    output_matrix_2 = np.zeros((8 , 4))
    
    output_matrix_2    = np.transpose(((1 * (input_matrix[ : , 0] == 0)) & (1 * (input_matrix[ : , 1] == 0)) , 
                          (1 * (input_matrix[ : , 0] == 0)) & (1 * (input_matrix[ : , 1] == 1)) , 
                          (1 * (input_matrix[ : , 0] == 1)) & (1 * (input_matrix[ : , 1] == 0)) , 
                          (1 * (input_matrix[ : , 0] == 1)) & (1 * (input_matrix[ : , 1] == 1))))
    
    output_matrix_3 = np.zeros((8 , 8))
    output_matrix_3 = np.transpose(( 
        1 * ((input_matrix[ : , 0] == 0) & (input_matrix[ : , 1] == 0) & (input_matrix[ : , 2] == 0)) , 
        1 * ((input_matrix[ : , 0] == 0) & (input_matrix[ : , 1] == 0) & (input_matrix[ : , 2] == 1)) , 
        1 * ((input_matrix[ : , 0] == 0) & (input_matrix[ : , 1] == 1) & (input_matrix[ : , 2] == 0)) , 
        1 * ((input_matrix[ : , 0] == 0) & (input_matrix[ : , 1] == 1) & (input_matrix[ : , 2] == 1)) , 
        
        1 * ((input_matrix[ : , 0] == 1) & (input_matrix[ : , 1] == 0) & (input_matrix[ : , 2] == 0)) , 
        1 * ((input_matrix[ : , 0] == 1) & (input_matrix[ : , 1] == 0) & (input_matrix[ : , 2] == 1)) , 
        1 * ((input_matrix[ : , 0] == 1) & (input_matrix[ : , 1] == 1) & (input_matrix[ : , 2] == 0)) , 
        1 * ((input_matrix[ : , 0] == 1) & (input_matrix[ : , 1] == 1) & (input_matrix[ : , 2] == 1)) , 
        ))
        
    return output_matrix_1 , output_matrix_2 , output_matrix_3

def initialize_weight_matrices(): 
    weight_matrix_1 = np.zeros((4 , 2))
    
    weight_matrix_2 = np.zeros((6 , 4))
    input_to_hidden = np.copy(weight_matrix_1)
    
    weight_matrix_3     = np.zeros((8 , 8))
    input_to_hidden = np.copy(weight_matrix_1) #This is from the input to the 2 units
    input_to_hidden_2   = weight_matrix_2[4 :  , : ] #This is from the 2 units to the 4 units
    
    return weight_matrix_1 , weight_matrix_2 , input_to_hidden , weight_matrix_3 , input_to_hidden_2 , input_to_hidden_2

#%% Define information used for all models

#Define the number of trials 
n_trials = 1000

input_matrix = general_initialization()

##Initialize all the output matrices (1 , 2 , 3 for the models)
output_matrix_1 , output_matrix_2 , output_matrix_3 = initialize_all_outputs()

##Initialize all the weight matrices
weight_matrix_1 , weight_matrix_2 , input_to_hidden , weight_matrix_3 , input_to_hidden_2 , input_to_hidden_2 = initialize_weight_matrices()

#%% Model 1
def train_model_1(input_matrix , weight_matrix_1 , output_matrix_1 , epoch , choice): 
    
    current_index = choice - epoch * 8
    
    #current index used to be i
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_1[current_index , : ]
    
    in_i = np.matmul(current_trial , weight_matrix_1)
    
    decision_model = softmax(in_i)
    
    error =  current_output - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    input_matrix_2 = current_trial[np.newaxis , : ]
    error_2 = error[np.newaxis , : ]
    
    ##And then update the weight matrix
    weight_matrix_1 += np.matmul(np.transpose(input_matrix_2) , error_2) 
        
    return weight_matrix_1 , decision_model , error , current_output 

#%% Model 2

def train_model_2(input_matrix , weight_matrix_1 , weight_matrix_2 , input_to_hidden , output_matrix_2 , epoch , choice): 
    
    input_to_hidden = np.copy(weight_matrix_1)
    
    current_index = choice - epoch * 8
    
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_2[current_index , : ]
    
    hidden = softmax(model_predict(current_trial , input_to_hidden))
    
    full_input = make_input_layer(current_trial , hidden)
        
    in_i = np.matmul(full_input , weight_matrix_2)
    
    decision_model = softmax(in_i)
    
    error = current_output - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    full_input = full_input[np.newaxis , : ]
    error_2 = error[np.newaxis , : ]
    
    ##Last 2 rows of weight_matrix_2 is then the hidden to out with 4 output nodes 
    weight_matrix_2 += np.matmul(np.transpose(full_input) , error_2)
        
    return weight_matrix_2 , decision_model , error ,  current_output

#%% Model 3

def train_model_3(input_matrix , weight_matrix_1 , weight_matrix_2 , weight_matrix_3 , input_to_hidden , output_matrix_3 , epoch , choice):
    
    input_to_hidden = np.copy(weight_matrix_1) #This is from the input to the 2 units
    input_to_hidden_2   = weight_matrix_2[4 :  , : ]     
    
    current_index = choice - epoch * 8
    
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_3[current_index , : ]
    
    hidden_1 = softmax(model_predict(current_trial , input_to_hidden))
    
    hidden_2 = softmax(model_predict(hidden_1 , input_to_hidden_2))
    
    full_input = make_input_layer(current_trial , hidden_2)
    
    in_i = np.matmul(full_input , weight_matrix_3)
    
    decision_model = softmax(in_i)
    
    error = current_output - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    error_2 = error[np.newaxis , : ]
    full_input = full_input[np.newaxis , : ]
    
    ##And then update the weight matrix
    weight_matrix_3 += np.matmul(np.transpose(full_input) , error_2)

    return weight_matrix_3 , decision_model , error , current_output

#%% Higher level that guides the decision process

import pandas as pd
import os

##For now, I've implemented 3 parameters (LP (signed), ULP (unsigned LP) and PE). 
##For now, I'll fix the weights at 1 and -1 (minus because of the reversed effect; i.e., lower PE is "better")
weight_LP   = 1
weight_ULP  = 1
weight_PE   = -1
weight_UPE  = -1

def get_parameters(data , choice): 
    ##The errors we base our decision on for the current trial depends on the error on the previous trial, so we get that
    error_1 = data.loc[choice - 1 , "error_1"]
    error_2 = data.loc[choice - 1 , "error_2"]
    error_3 = data.loc[choice - 1 , "error_3"]
    
    ##The LP we base our decision on is the one from the previous trial
    #If we didn't choose the model, we get the LP from the previous trial, if we did choose it, we calculate the LP
    #To calculate the LP on (t - 1), we take the difference in error between (t - 1) and (t - 2)
    if choice != 1: 
        if data.loc[choice - 1, "error_1"] != data.loc[choice - 2 , "error_1"]: 
            LP_1 = data.loc[choice - 2 , "error_1"] - data.loc[choice - 1 , "error_1"]
            LP_2 = data.loc[choice - 1 , "LP_2"]
            LP_3 = data.loc[choice - 1 , "LP_3"]
        elif data.loc[choice - 1 , "error_2"] != data.loc[choice - 2 , "error_2"]: 
            LP_2 = data.loc[choice - 2 , "error_2"] - data.loc[choice - 1 , "error_2"]
            LP_1 = data.loc[choice - 1 , "LP_1"]
            LP_3 = data.loc[choice - 1 , "LP_3"]
        elif data.loc[choice - 1 , "error_3"] != data.loc[choice - 2 , "error_3"]: 
            LP_3 = data.loc[choice - 2 , "error_3"] - data.loc[choice - 1 , "error_3"]
            LP_1 = data.loc[choice - 1 , "LP_1"]
            LP_2 = data.loc[choice - 1 , "LP_2"]
    else: 
        LP_1 = 0 
        LP_2 = 0
        LP_3 = 0
        
    return LP_1 , LP_2 , LP_3 , error_1 , error_2 , error_3

def save_variables(): 
    ##Get some data in the file
    data.loc[choice , "ULP_1"] = np.abs(data.loc[choice , "LP_1"])
    data.loc[choice , "ULP_2"] = np.abs(data.loc[choice , "LP_2"])
    data.loc[choice , "ULP_3"] = np.abs(data.loc[choice , "LP_3"])
    
    data.loc[choice , "UPE_1"] = np.abs(data.loc[choice , "error_1"])
    data.loc[choice , "UPE_2"] = np.abs(data.loc[choice , "error_2"])
    data.loc[choice , "UPE_3"] = np.abs(data.loc[choice , "error_3"])

data = pd.DataFrame()

n_epochs = 10

n_choices = input_matrix.shape[0] * n_epochs
choices = [1 , 2 , 3]
epoch = 0

MSE = []
MSE_1 = []
MSE_2 = []
MSE_3 = []

LP_1 = 0 
LP_2 = 0
LP_3 = 0

for choice in range(n_choices): 
    
    if choice % 8 == 0: 
        epoch += 1
    
    if choice == 0: 
        ##If it's the first trial, then there's no information yet on how to decide, so we go random 
        model_choice = np.random.choice(choices)
        
        data.loc[choice , "model_choice"] = model_choice
        
        ##And in the first trial, there's no LP yet, because we need the PE of the current and previous trial.         
        data.loc[choice , ["LP_1" , "LP_2" , "LP_3"]] = [0 , 0 , 0]
        data.loc[choice + 1 , ["LP_1" , "LP_2" , "LP_3"]] = [0 , 0 , 0]
        
        ##Errors still change, but I want them in order, so that's why I do them now once here. 
        data.loc[choice , ["error_1" , "error_2" , "error_3"]] = [0 , 0 , 0]
        
    else: 
        
        LP_1 , LP_2 , LP_3 , error_1 , error_2 , error_3 = get_parameters(data , choice)
        
        ULP_1 = np.abs(LP_1)
        ULP_2 = np.abs(LP_2)
        ULP_3 = np.abs(LP_3)
        
        UPE_1 = np.abs(error_1)
        UPE_2 = np.abs(error_2)
        UPE_3 = np.abs(error_3)
        
        #Get the values for the choice for each model
        model_1 = np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1 + weight_UPE * UPE_1) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1 + weight_UPE * UPE_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2 + weight_UPE * UPE_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3 + weight_UPE * UPE_3))
        model_2 = np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2 + weight_UPE * UPE_2) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1 + weight_UPE * UPE_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2 + weight_UPE * UPE_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3 + weight_UPE * UPE_3))
        model_3 = np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3 + weight_UPE * UPE_3) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1 + weight_UPE * UPE_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2 + weight_UPE * UPE_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3 + weight_UPE * UPE_3))
        
        choosing_model = [model_1 , model_2 , model_3]
        
        #Select a model; do "+1" to get the choice to 1, 2 or 3 instead of zero-based index 
        model_choice = np.argmax(choosing_model) + 1 
        data.loc[choice , "model_choice"] = model_choice
        
    if model_choice == 1: 
        #Train model 1
        weight_matrix_1 , decision_model , error , output_matrix = train_model_1(input_matrix , weight_matrix_1 , output_matrix_1 , epoch , choice)
        
        #Get the error in the file
        data.loc[choice , "error_1"] = sum(error ** 2)
        
        if choice != 0: 
            #Only if not the first trial, because otherwise I can't go one trial back ofccccc
            #Basically just copy the error from the last trail to the current one for the options not chosen. 
            data.loc[choice , "error_2"] = data.loc[choice - 1 , "error_2"]
            data.loc[choice , "error_3"] = data.loc[choice - 1 , "error_3"]
    
            data.loc[choice , "LP_1"] = data.loc[choice - 1 , "error_1"] - data.loc[choice , "error_1"]
            data.loc[choice , "LP_2"] = data.loc[choice - 1 , "LP_2"]
            data.loc[choice , "LP_3"] = data.loc[choice - 1 , "LP_3"]
            
        else: 
            #On the first trial, I put the errors to -1 (arbitrary, so if I need to change this, I can). 
            data.loc[choice , "error_2"] = -1
            data.loc[choice , "error_3"] = -1
        
        MSE_1.append(- sum(np.log(decision_model) * output_matrix))
            
    ##And then do the same for the other models (if they get chosen)
    elif model_choice == 2: 
        weight_matrix_2 , decision_model , error , output_matrix= train_model_2(input_matrix , weight_matrix_1 , weight_matrix_2 , input_to_hidden , output_matrix_2 , epoch , choice)
        
        data.loc[choice , "error_2"] = sum(error ** 2)
        
        if choice != 0: 
            data.loc[choice , "error_1"] = data.loc[choice - 1 , "error_1"]
            data.loc[choice , "error_3"] = data.loc[choice - 1 , "error_3"]
            
            data.loc[choice , "LP_2"] = data.loc[choice - 1 , "error_2"] - data.loc[choice , "error_2"]
            data.loc[choice , "LP_1"] = data.loc[choice - 1 , "LP_1"]
            data.loc[choice , "LP_3"] = data.loc[choice - 1 , "LP_3"]
            
        else: 
            data.loc[choice , "error_1"] = -1
            data.loc[choice , "error_3"] = -1
            
        MSE_2.append(- sum(np.log(decision_model) * output_matrix))
        
    else: 
        weight_matrix_3 , decision_model , error , output_matrix= train_model_3(input_matrix , weight_matrix_1 , weight_matrix_2 , weight_matrix_3 , input_to_hidden , output_matrix_3 , epoch , choice)
        
        data.loc[choice , "error_3"] = sum(error ** 2)
        
        if choice != 0: 
            data.loc[choice , "error_1"] = data.loc[choice - 1 , "error_1"]
            data.loc[choice , "error_2"] = data.loc[choice - 1 , "error_2"]
            
            data.loc[choice , "LP_3"] = data.loc[choice - 1 , "error_3"] - data.loc[choice , "error_3"]
            data.loc[choice , "LP_1"] = data.loc[choice - 1 , "LP_1"]
            data.loc[choice , "LP_2"] = data.loc[choice - 1 , "LP_2"]
            
        else: 
            data.loc[choice , "error_1"] = -1
            data.loc[choice , "error_2"] = -1
        
        MSE_3.append(- sum(np.log(decision_model) * output_matrix))
    
    MSE.append(- sum(np.log(decision_model) * output_matrix))
    
    save_variables()


"""
New_MSE = []

for i in range(len(MSE)): 
    
    New_MSE.append(np.mean(MSE[ : i + 1]))

MSE[n_choices - 50 : n_choices]
"""

from matplotlib import pyplot as plt


fig , ax = plt.subplots(2 , 2)

ax[0 , 0].plot(MSE_1)
ax[0 , 1].plot(MSE_2)
ax[1 , 0].plot(MSE_3)
#ax[1 , 1].plot(MSE)

len(MSE_1)

#plt.plot(MSE)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#plt.plot(moving_average(MSE_1 , n = 10))
#plt.plot(MSE_1)

ax[1 , 1].plot(moving_average(MSE_1 , n = 10))
fig.show()

#plt.plot(MSE_1)

#plt.plot(New_MSE)

##And then save it
#os.chdir("/Users/wardclaeys/OneDrive - UGent/Modelling")


data.to_csv('data/dataframe_1.csv')

#weight_matrix_1


"""

learning progress as the decrease in PE (signed)
unisgned LP is just the change in PEs (can be both positive or negative)


LP  = (x̂_𝑡−1 - x_𝑡−1) – (x̂_𝑡 - x_𝑡)
    = PE_t-1 - PE_t

PE => PEt = PEt-1 + alpha * PEt
Novelty; in Poli et al (2022) s(-t) 

"where t indicates the overall number of trials a given character has been observed, 
and s is a smooth function. Given that we did not have a-priori assumptions on the rate 
of change in novelty or random search as a function of time, we used additive terms instead 
of, for instance, a logarithmic or exponential function. "

"""

"""
##Get the index of the last trial where a certain model was selected (this to use for the LP update)
if choice != 0 and choice != 1: 
    if (data.loc[choice , "error_1"] == data.loc[choice - 1 , "error_1"]) and (data.loc[choice - 1 , "error_1"] != data.loc[choice - 2 , "error_1"]): 
        index_model_1 = choice
    if (data.loc[choice , "error_2"] == data.loc[choice - 1 , "error_2"]) and (data.loc[choice - 1 , "error_2"] != data.loc[choice - 2 , "error_2"]): 
        index_model_2 = choice
    if (data.loc[choice , "error_3"] == data.loc[choice - 1 , "error_3"]) and (data.loc[choice - 1 , "error_3"] != data.loc[choice - 2 , "error_3"]): 
        index_model_3 = choice
"""


"""
Calculate MSE
Go like this: 
    
    summation of all previous trials and then multiply the target values with the natural log of the decision of the model

"""

#output_matrix_1[4 , : ]
#softmax(np.matmul(input_matrix[4 , : ] , weight_matrix_1))




