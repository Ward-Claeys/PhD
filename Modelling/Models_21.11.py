#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION OF NOVEMBER 21

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

def initialize_model_2(): 
    weight_matrix_2 = np.zeros((5 , 4))
    input_to_hidden = np.copy(weight_matrix_1)
    
    return weight_matrix_2 , input_to_hidden

def initialize_model_3(): 
    
    weight_matrix_3     = np.zeros((7 , 8))
    input_to_hidden = np.copy(weight_matrix_1) #This is from the input to the 2 units
    input_to_hidden_2   = weight_matrix_2[3 :  , : ] #This is from the 2 units to the 4 units
    
    return weight_matrix_3 , input_to_hidden , input_to_hidden_2

#%% Define information used for all models

#Define the number of trials 
n_trials = 1000

input_matrix , participant_orientation , participant_size = general_initialization()

##Initialize all the output matrices (1 , 2 , 3 for the models)
output_matrix_1 , output_matrix_2 , output_matrix_3 = initialize_all_outputs()

#%% Model 1

#Initialize weight matrix
weight_matrix_1 = np.zeros((3 , 2))

for i in range(n_trials):
    current_trial = input_matrix[i , : ]
    current_output = output_matrix_1[i , : ]
    
    in_i = np.matmul(input_matrix[i , : ] , weight_matrix_1)
    
    decision_model = softmax(in_i)
    
    error =  current_output  - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    input_matrix_2 = current_trial[np.newaxis , : ]
    error_2 = error[np.newaxis , : ]
    
    ##And then update the weight matrix
    weight_matrix_1 += np.matmul(np.transpose(input_matrix_2) , error_2)

#%% Model 2

weight_matrix_2 , input_to_hidden = initialize_model_2()

for i in range(n_trials):
    current_trial = input_matrix[i , : ]
    current_output = output_matrix_2[i , : ]
    
    hidden = model_predict(input_matrix[i , : ] , input_to_hidden)
    
    full_input = make_input_layer(current_trial , hidden)
        
    in_i = np.matmul(full_input , weight_matrix_2)
    
    decision_model = softmax(in_i)
    
    error = current_output - decision_model
    
    ##Make it ready to matmul (otherwise not possible due to incompatibility)
    full_input = full_input[np.newaxis , : ]
    error_2 = error[np.newaxis , : ]
    
    ##Last 2 rows of weight_matrix_2 is then the hidden to out with 4 output nodes 
    weight_matrix_2 += np.matmul(np.transpose(full_input) , error_2)

#%% Model 3

weight_matrix_3 , input_to_hidden , input_to_hidden_2 = initialize_model_3()

for i in range(n_trials):
    current_trial = input_matrix[i , : ]
    current_output = output_matrix_3[i , : ]
    
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


