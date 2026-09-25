#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:06:27 2026

@author: wardclaeys
"""

# Import modules
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize
from Model_simulation_Shared_boundaries import generate_dataset

# Avoid warnings
import warnings

warnings.filterwarnings("ignore")

os.chdir("/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery")

def sigmoid(x, slope , shift):
    return 1 / (1 + np.exp(-slope * x + shift * slope))

def stay_leave_decision(boundaries , current_task , RAs):

    if current_task == 1: 
        #Get the normalized value for the decision
        
        probability = sigmoid(RAs[0] , boundaries[2] , boundaries[0])
        
        #If the prob is high, then that's a sign to move on to a more difficult task, if not, we stay 
        #The first prob is the decrement one (which is not possible, since it's the easiest task), the second is the stay one and the third is the increment one 
        
        stay_leave = [0 , 1 - probability , probability]
        #prob = RAs[0] - noise 
        #stay_leave = [0 , 1 - prob , prob]
        
    elif current_task == 2: 
        #Get the normalized value for the decision
        #Now we have a probability to increment  
        increment = sigmoid(RAs[1] , boundaries[2] , boundaries[0])
        #If the prob is high, then that's a sign to move on to a more difficult task, if not, we stay or decrement to easier task
        
        #Then we also calculate the prob to stay or decrement
        decrement_stay = sigmoid(RAs[1] , boundaries[2] , boundaries[1])
    
        decrement_stay = [1 - decrement_stay , decrement_stay]
        decrement_stay = [x * (1 - increment) for x in decrement_stay]
        
        #So now we have a list with 3 probabilities; a prob to decrement, one to stay and one to increment 
        stay_leave = [decrement_stay[0] , decrement_stay[1] , increment]
    
    else: 
        #Get the normalized value for the decision
        probability = sigmoid(RAs[2] , boundaries[2] , boundaries[1])
        
        #If the prob is high, then that's a sign to stay as this is the last task
        #The first prob is the decrement one and the second is the stay one (here reversed because a high one is a stay, cuz you're at the boundary), the last is increment, but zero since not possible as it's the most difficult task 
        stay_leave = [1 - probability , probability , 0]
        
        #prob = RAs[2] - noise
        #stay_leave = [0 , 1 - prob , prob]
        #print(probability)
        
    return stay_leave

# Likelihood function for empirical data
#It only gets calculated for trials where participants do a meta-decision here, not for all trials 
def likelihood(parameter_set, data):

    df = pd.read_csv(data)  # Read data
    
    #df = df.iloc[indices]
    
    ntrials = df.shape[0]  # Extract number of trials

    # Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0  # this is calculated by summing over trials the log( L(parameter set|response on that trial) )
    
    meta_trials = 0 
    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
        
        #The first trial, we can't yet increase, decrease or stay, so skip that one 
        
        if (trial != 0): #ntrials: 
            
            #Only relevant trials are the decision trials, so the meta-trials 
            #if df.loc[trial , "Meta_trial"] == 1: 
            #(trial + 1) in indices: 
                
            meta_trials += 1 
            estimated_data.loc[simulation , "n_trials"] = meta_trials
            
            #Previous trial as this information is what is available to participants to make a decision. 
            loglikelihoods = np.log(stay_leave_decision(boundaries = parameter_set , current_task = df.loc[trial - 1 , "Current_task"] , RAs = [df.loc[trial - 1 , "accuracy_1"] , df.loc[trial - 1 , "accuracy_2"] , df.loc[trial - 1 , "accuracy_3"]])) 
            
            #So we take the current castle - the previous one
            #I ask for the sign of this. If decrement; we get a negative value, so this becomes -1 (no matter how much decrement), 0 if stay and +1 if increment (does not matter how much)
            #Then we add one to make indices between 0 and 2 out of them
            response = int(np.sign(df.loc[trial  , "Current_task"] - df.loc[trial - 1 , "Current_task"]) + 1)
            
            # then select the probability of the actual response given the parameter set
            current_loglikelihood = loglikelihoods[response]#Check if positice!! 

            # Add L(parameter set|current response) to the total log likelihood
            summed_logL = summed_logL + current_loglikelihood
        
    return -summed_logL 

#Define columns for output file
column_list = ["simulation" , "Boundary_Decrement", "Boundary_Increment" , "AIC" , "n_trials" , "BIC" , "negLL" , "Estimated_slope" , "Estimated_Boundary_Increment", "Estimated_Boundary_Decrement" , "slope" , "n_trials_1" , "n_trials_2" , "n_trials_3"]
estimated_data = pd.DataFrame(columns=column_list)

column_list_2 = ["Current_task" , "Accuracy" , "Boundary_Increase" , "Boundary_Decrease" , 
               "Meta_trial" , "Probability_correct_1" , "Probability_correct_2" , "Probability_correct_3" , "Decision" , "Prob_Dec" , "Prob_Stay" , "Prob_Inc" , 
               "accuracy_1" , "accuracy_2" , "accuracy_3"]

n_simulations = 100
n = 100

for simulation in range(n_simulations): 
    
    dataset = generate_dataset(n_trials = n) 
    dataset.to_csv("Simulated_data.csv", columns = column_list_2 , float_format ='%.3f')
    
    start_params = np.random.uniform(0 , 1) , np.random.uniform(0 , 1) , np.random.uniform(0 , 1) 

    optimization_output = optimize.minimize(fun = likelihood , x0 = start_params , args = "Simulated_data.csv" , options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0} , method = "Nelder-Mead")# , verbose = 1)
    
    #Get minimum log likelihood and parameter estimations
    LL = optimization_output['fun']
    estimated_parameters = optimization_output['x']
    
    AIC = - 2 * -LL + 2 * len(start_params)
    BIC = - 2 * -LL + len(start_params) * np.log(n)
    #BIC = 0
    
    print(LL)
    
    #print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
    #Store everything in output file
    #estimated_data.loc[idx , ["Estimated_PE", "Estimated_LP" , "Estimated_Nov", "Negative_LogL"]] = [estimated_parameters[0] , estimated_parameters[1] , estimated_parameters[2] , LL]
    estimated_data.loc[simulation , ["simulation" , "Estimated_Boundary_Increment" , "Estimated_Boundary_Decrement" , "AIC" , "BIC" , "negLL" , "Estimated_slope"]] = [simulation , estimated_parameters[0] , estimated_parameters[1] , AIC , BIC , LL , estimated_parameters[2]]
    
    estimated_data.loc[simulation , ["Boundary_Increment" , "Boundary_Decrement" , "slope"]] = [dataset.loc[1 , "Boundary_Increase"] , dataset.loc[1 , "Boundary_Decrease"] , dataset.loc[1 , "slope"]]

    estimated_data.loc[simulation , ["n_trials_1" , "n_trials_2" , "n_trials_3"]] = [sum(dataset.loc[ : , "Current_task"] == 1) , sum(dataset.loc[ : , "Current_task"] == 2) , sum(dataset.loc[ : , "Current_task"] == 3)]

    print("Simulated data")

#Write results of parameter fitting
estimated_data.to_csv("Recovery_Shared_boundary.csv", columns = column_list, float_format ='%.3f')
print("End of fitting procedure")
