#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:51:33 2026

@author: wardclaeys
"""

# Import modules
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize
from Parameter_recovery import generate_dataset

# Avoid warnings
import warnings

warnings.filterwarnings("ignore")

os.chdir("/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery")

def softmax(values = np.array([0.5]) , PEs = np.array([0.5 , 0.5 , 0.5]) , LPs = np.array([0.5 , 0.5 , 0.5])  , Novs = np.array([0.5 , 0.5 , 0.5])):
    
    nov_1 = np.exp(- Novs[0])
    nov_2 = np.exp(- Novs[1])
    nov_3 = np.exp(- Novs[2])
    
    numerator_1 = np.exp(values[0] * PEs[0] + values[1] * LPs[0] + values[2] * nov_1)
    numerator_2 = np.exp(values[0] * PEs[1] + values[1] * LPs[1] + values[2] * nov_2)
    numerator_3 = np.exp(values[0] * PEs[2] + values[1] * LPs[2] + values[2] * nov_3)
    
    denom = np.sum([numerator_1 , numerator_2 , numerator_3])
    
    response_probabilities = [numerator_1 / denom, numerator_2 / denom, numerator_3 / denom]
    
    #response_probabilities = np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs) / np.sum(np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs))
    
    return response_probabilities
    
# Likelihood function for empirical data
def likelihood(parameter_set, data):

    df = pd.read_csv(data)  # Read data
    
    df = df.query("Chosen_castle != -9999")
    
    ntrials = df.shape[0]  # Extract number of trials

    # Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0  # this is calculated by summing over trials the log( L(parameter set|response on that trial) )
    
    #Initialize the moving window to calculate the confidence of participants 
    moving_window_1 = []
    moving_window_2 = []
    moving_window_3 = []
    
    novelty_1 , novelty_2 , novelty_3 = 0 , 0 , 0 
    
    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials - 30):
        #I will go for the probability to be correct as the mean of the last 10 trials
        #LP as the change with the new trial in there 
        #nov just the same as else 
        
    #Define the variables that are important for the likelihood estimation process
        response = int(df.loc[trial , "Chosen_castle"]) - 1 #the stimulus shown on this trial
        
        
        if trial == 0: 
            
            PE_1 , PE_2 , PE_3 = 0 , 0 , 0 
            LP_1 , LP_2 , LP_3 = 0 , 0 , 0 
            
            if response == 0: 
                novelty_1 += 1
            elif response == 1: 
                novelty_2 += 1 
            else: 
                novelty_3 += 1
            
            PEs = np.array([PE_1 , PE_2 , PE_3])
            LPs = np.array([LP_1 , LP_2 , LP_3])
            Novs = np.array([novelty_1 , novelty_2 , novelty_3])
            
        else: 
            
            ##Get the previous ones here for the estimation 
            PEs = np.array([PE_1 , PE_2 , PE_3])
            LPs = np.array([LP_1 , LP_2 , LP_3])
            Novs = np.array([novelty_1 , novelty_2 , novelty_3])
            
            #Then get the current ones 
            
            #Append the accuracy of the current trial in a specific condition
            if response == 0: 
                moving_window_1.append(int(df.loc[trial , "Accuracy"]))
                probability_1 = np.mean(moving_window_1[-10 : ])
                
                PE_1 = int(df.loc[trial , "Accuracy"]) - probability_1
                
                novelty_1 += 1 
                novelty_2 , novelty_3 = 0 , 0 
                
                if trial != 1: 
                    LP_1 = df.loc[trial - 2 , "PE_1"] - df.loc[trial - 1 , "PE_1"]
                
            elif response == 1: 
                moving_window_2.append(int(df.loc[trial , "Accuracy"]))
                probability_2 = np.mean(moving_window_2[-10 : ])
                
                PE_2 = int(df.loc[trial , "Accuracy"]) - probability_2
                
                novelty_2 += 1 
                novelty_1 , novelty_3 = 0 , 0 
                
                if trial != 1: 
                    LP_2 = df.loc[trial - 2 , "PE_2"] - df.loc[trial - 1 , "PE_2"]
                
            else: 
                moving_window_3.append(int(df.loc[trial , "Accuracy"]))
                probability_3 = np.mean(moving_window_3[-10 : ])
                
                PE_3 = int(df.loc[trial , "Accuracy"]) - probability_3
                
                novelty_3 += 1 
                novelty_2 , novelty_2 = 0 , 0 
                
                if trial != 1: 
                    LP_3 = df.loc[trial - 2 , "PE_3"] - df.loc[trial - 1 , "PE_3"]
            
        df.loc[trial , ["PE_1" , "PE_2" , "PE_3"]] = [PE_1 , PE_2 , PE_3]
        df.loc[trial , ["LP_1" , "LP_2" , "LP_3"]] = [LP_1 , LP_2 , LP_3]
        df.loc[trial , ["Nov_1" , "Nov_2" , "Nov_3"]] = [novelty_1 , novelty_2 , novelty_3]
        
        loglikelihoods = np.log(softmax(parameter_set , PEs , LPs , Novs)) 

        # then select the probability of the actual response given the parameter set
        current_loglikelihood = loglikelihoods[response]

        # Add L(parameter set|current response) to the total log likelihood
        summed_logL = summed_logL + current_loglikelihood

    return -summed_logL

#Define columns for output file
column_list = ["Current_PE", "Current_LP" , "Current_Nov", "Estimated_PE", "Estimated_LP" , "Estimated_Nov", "Negative_LogL"]
estimated_data = pd.DataFrame(columns=column_list)

PE_values = np.random.uniform(-1 , 0 , 1000)
np.random.shuffle(PE_values)
LP_values = np.random.uniform(0 , 1 , 1000)
np.random.shuffle(LP_values)
Nov_values = np.random.uniform(0 , 1 , 1000)
np.random.shuffle(Nov_values)

parameter_values = np.column_stack([PE_values , LP_values , Nov_values])


if __name__ == '__main__':
    #Extract path to data folder
    
    directory = "/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery/Pilot_17th_March"

    #Get a list of files in the folder and filter on csv files that are not previous fitting results
    filelist = os.listdir(directory)
    filtered_filelist = [x for x in filelist if x[-3::]=="csv"]
    filtered_filelist = [x for x in filtered_filelist if x != "Fitting_results.csv"]
    filtered_filelist = [x for x in filtered_filelist if x[-13::] != "Simulated.csv"]

    #Go to that directory
    os.chdir(directory)

    #Define the starting parameters for the likelihood
    start_params = np.random.uniform(0 , 1) , np.random.uniform(0 , 1) , np.random.uniform(0 , 1)
    
    idx = -1
    for file in filtered_filelist:
        idx +=1
        
        #Get subject id 
        sub = file[2 : -4]
        
        #Minimize negative loglikelihood
        #optimization_output = optimize.minimize(fun = likelihood , x0 = [start_params, df]) #, options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})
        
        optimization_output = optimize.minimize(fun = likelihood , x0 = start_params , args =(tuple([file])) , options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})
        
        #Get minimum log likelihood and parameter estimations
        LL = optimization_output['fun']
        estimated_parameters = optimization_output['x']
        
        #print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
        #Store everything in output file
        #estimated_data.loc[idx , ["Estimated_PE", "Estimated_LP" , "Estimated_Nov", "Negative_LogL"]] = [estimated_parameters[0] , estimated_parameters[1] , estimated_parameters[2] , LL]
        estimated_data.loc[sub , ["sub_id" , "Estimated_PE" , "Estimated_LP" , "Estimated_Nov"]] = [sub , estimated_parameters[0] , estimated_parameters[1] , estimated_parameters[2]]
        
        print("Simulated data")

    #Write results of parameter fitting
estimated_data.to_csv("Fitting_results_participants.csv", columns = column_list, float_format ='%.3f')
print("End of fitting procedure")



