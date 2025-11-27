#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 12:57:01 2025

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
    
    ##Get the different possible inputs
    orientation = np.arange(start = 0 , stop = 1 , step = 0.025)
    size        = np.arange(start = 0 , stop = 1 , step = 0.1)
    color       = np.arange(start = 0 , stop = 1 , step = 0.04)
    
    #get random participant orientation and size
    participant_orientation = np.random.choice(orientation)

    participant_size = 0
    while (participant_size < 0.25) or (participant_size > 0.75): 
        participant_size = np.random.choice(size)
    
    ##Create the input matrix
    input_matrix = np.zeros((n_trials , 3))
    
    ##And fill it
    input_matrix[ : , 0]    = np.random.choice(orientation , size = n_trials , replace = True)
    input_matrix[ : , 1]    = np.random.choice(size , size = n_trials , replace = True)
    input_matrix[ : , 2]    = np.random.choice(color , size = n_trials , replace = True)
        
    return input_matrix , participant_orientation , participant_size

def initialize_all_outputs(): 
    output_matrix_1 = np.zeros((n_trials , 2))
    ##Create the output matrix [1 , 0] if current rotation is smaller than the participant one and [0 , 1] if larger
    output_matrix_1[ : , : ] = np.transpose([1 * (input_matrix[ : , 0] < participant_orientation) , 1 - (1 * (input_matrix[ : , 0] < participant_orientation))]) 

    output_matrix_2 = np.zeros((n_trials , 4))
    output_matrix_2 = np.transpose(np.array((1 * ((input_matrix[ : , 0] <= participant_orientation) & (input_matrix[ : , 1] <= participant_size)) , 
    1 * ((input_matrix[ : , 0] < participant_orientation) & (input_matrix[ : , 1] > participant_size)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] < participant_size)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] > participant_size)))))
    
    output_matrix_3 = np.zeros((n_trials , 8))
    output_matrix_3 = np.transpose(np.array((
    1 * ((input_matrix[ : , 0] <= participant_orientation) & (input_matrix[ : , 1] <= participant_size) & (input_matrix[ : , 2] <= 0.5)) ,
    1 * ((input_matrix[ : , 0] < participant_orientation) & (input_matrix[ : , 1] < participant_size) & (input_matrix[ : , 2] > 0.5)) , 
    1 * ((input_matrix[ : , 0] < participant_orientation) & (input_matrix[ : , 1] > participant_size) & (input_matrix[ : , 2] < 0.5)) , 
    1 * ((input_matrix[ : , 0] < participant_orientation) & (input_matrix[ : , 1] > participant_size) & (input_matrix[ : , 2] > 0.5)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] < participant_size) & (input_matrix[ : , 2] < 0.5)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] < participant_size) & (input_matrix[ : , 2] > 0.5)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] > participant_size) & (input_matrix[ : , 2] < 0.5)) , 
    1 * ((input_matrix[ : , 0] > participant_orientation) & (input_matrix[ : , 1] > participant_size) & (input_matrix[ : , 2] > 0.5)))))
    
    return output_matrix_1 , output_matrix_2 , output_matrix_3

def initialize_weight_matrices(): 
    weight_matrix_1 = np.zeros((3 , 2))
    
    weight_matrix_2 = np.zeros((5 , 4))
    input_to_hidden = np.copy(weight_matrix_1)
    
    weight_matrix_3     = np.zeros((7 , 8))
    input_to_hidden = np.copy(weight_matrix_1) #This is from the input to the 2 units
    input_to_hidden_2   = weight_matrix_2[3 :  , : ] #This is from the 2 units to the 4 units
    
    return weight_matrix_1 , weight_matrix_2 , input_to_hidden , weight_matrix_3 , input_to_hidden_2 , input_to_hidden_2

#%% Define information used for all models

#Define the number of trials 
n_trials = 1000

input_matrix , participant_orientation , participant_size = general_initialization()

##Initialize all the output matrices (1 , 2 , 3 for the models)
output_matrix_1 , output_matrix_2 , output_matrix_3 = initialize_all_outputs()

##Initialize all the weight matrices
weight_matrix_1 , weight_matrix_2 , input_to_hidden , weight_matrix_3 , input_to_hidden_2 , input_to_hidden_2 = initialize_weight_matrices()

#%% Model 1
def train_model_1(input_matrix , weight_matrix_1): 
    
#    for i in range(input_matrix.shape[0]):
    current_index = np.random.choice(np.arange(input_matrix.shape[0]))
    
    #current index used to be i
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_1[current_index , : ]
    
    in_i = np.matmul(current_trial , weight_matrix_1)
    
    decision_model = softmax(in_i)
    
    error =  current_output  - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    input_matrix_2 = current_trial[np.newaxis , : ]
    error_2 = error[np.newaxis , : ]
    
    ##And then update the weight matrix
    weight_matrix_1 += np.matmul(np.transpose(input_matrix_2) , error_2) 
        
    return weight_matrix_1 , decision_model , error , current_output

#weight_matrix_1 = train_model_1(input_matrix , weight_matrix_1)

#%% Model 2

def train_model_2(input_matrix , weight_matrix_1 , weight_matrix_2 , input_to_hidden): 
    
    input_to_hidden = np.copy(weight_matrix_1)
    
#    for i in range(input_matrix.shape[0]):
    
    current_index = np.random.choice(np.arange(input_matrix.shape[0]))
    
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_2[current_index , : ]
    
    hidden = model_predict(current_trial , input_to_hidden)
    
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

#weight_matrix_2 = train_model_2(input_matrix , weight_matrix_1 , weight_matrix_2 , input_to_hidden)

#%% Model 3

def train_model_3(input_matrix , weight_matrix_1 , weight_matrix_2 , weight_matrix_3 , input_to_hidden):
    
    input_to_hidden = np.copy(weight_matrix_1) #This is from the input to the 2 units
    input_to_hidden_2   = weight_matrix_2[3 :  , : ]     
    
#    for i in range(input_matrix.shape[0]):
    
    current_index = np.random.choice(np.arange(input_matrix.shape[0]))
    
    current_trial = input_matrix[current_index , : ]
    current_output = output_matrix_3[current_index , : ]
    
    hidden_1 = model_predict(current_trial , input_to_hidden)
    
    hidden_2 = model_predict(hidden_1 , input_to_hidden_2)
    
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


#weight_matrix_3 = train_model_3(input_matrix , weight_matrix_1 , weight_matrix_2 , weight_matrix_3 , input_to_hidden)

#%% Just playing around with some paramters (as one does xp)

import pandas as pd
import os

##For now, I've implemented 3 parameters (LP (signed), ULP (unsigned LP) and PE). 
##For now, I'll fix the weights at 1 and -1 (minus because of the reversed effect; i.e., lower PE is "better")
weight_LP   = 1
weight_ULP  = 1
weight_PE   = -1

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

data = pd.DataFrame()

n_choices = 100
choices = [1 , 2 , 3]

for choice in range(n_choices): 
    
    if choice == 0: 
        ##If it's the first trial, then there's no information yet on how to decide, so we go random 
        model_choice = np.random.choice(choices)
        
        ##And in the first trial, there's no LP yet, because we need the PE of the current and previous trial. 
        data.loc[choice , "LP_1"] = 0
        data.loc[choice , "LP_2"] = 0
        data.loc[choice , "LP_3"] = 0
        
        data.loc[choice + 1 , "LP_1"] = 0
        data.loc[choice + 1 , "LP_2"] = 0
        data.loc[choice + 1 , "LP_3"] = 0

    else: 
        
        LP_1 , LP_2 , LP_3 , error_1 , error_2 , error_3 = get_parameters(data , choice)
        
        ULP_1 = np.abs(LP_1)
        ULP_2 = np.abs(LP_2)
        ULP_3 = np.abs(LP_3)
        
        #Get the values for the choice for each model
        model_1 = np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3))
        model_2 = np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3))
        model_3 = np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3) / (np.exp(weight_PE * error_1 + weight_LP * LP_1 + weight_ULP * ULP_1) + np.exp(weight_PE * error_2 + weight_LP * LP_2 + weight_ULP * ULP_2) + np.exp(weight_PE * error_3 + weight_LP * LP_3 + weight_ULP * ULP_3))
        
        choosing_model = [model_1 , model_2 , model_3]
        
        #Select a model; do "+1" to get the choice to 1, 2 or 3 instead of zero-based index 
        model_choice = np.argmax(choosing_model) + 1 
    
    #Put all the data in the file (I did the errors as well, but they get changed to actual values, I just wanted to get them in order in the file, not always a random order based on the choice of the model). 
    data.loc[choice , "model_choice"] = model_choice
    data.loc[choice , ["error_1" , "error_2" , "error_3"]] = [0 , 0 , 0]
    
    if model_choice == 1: 
        #Train model 1
        weight_matrix_1 , decision_model , error , output_matrix = train_model_1(input_matrix , weight_matrix_1)
        
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
            
    ##And then do the same for the other models (if they get chosen)
    elif model_choice == 2: 
        weight_matrix_2 , decision_model , error , output_matrix = train_model_2(input_matrix , weight_matrix_1 , weight_matrix_2 , input_to_hidden)
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
    else: 
        weight_matrix_3 , decision_model , error , output_matrix = train_model_3(input_matrix , weight_matrix_1 , weight_matrix_2 , weight_matrix_3 , input_to_hidden)
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
    
    ##Get some data in the file
    data.loc[choice , "ULP_1"] = np.abs(data.loc[choice , "LP_1"])
    data.loc[choice , "ULP_2"] = np.abs(data.loc[choice , "LP_2"])
    data.loc[choice , "ULP_3"] = np.abs(data.loc[choice , "LP_3"])

##And then save it
data.to_csv('data/dataframe_1.csv')

#os.chdir("/Users/wardclaeys/OneDrive - UGent/Modelling")

"""

learning progress as the decrease in PE (signed)
unisgned LP is just the change in PEs (can be both positive or negative)


LP  = (x̂_𝑡−1 - x_𝑡−1) – (x̂_𝑡 - x_𝑡)
    = PE_t-1 - PE_t

PE => PEt = PEt-1 + alpha * PEt

PC = 1/𝑛 ∑_(𝑡^′=𝑡−𝑛)^𝑡▒𝑦_𝑡′ 
=> PC = (1/n) sommatie van (tijd - n) tot (tijd) y_t
=> Dus n bepalen en dan het gemiddelde berekenen over die laatste hoeveelheid trials

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











