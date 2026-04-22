#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:45:04 2026

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

    df = pd.read_csv("Check_this_file.csv")  # Read data
    ntrials = df.shape[0]  # Extract number of trials

    # Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0  # this is calculated by summing over trials the log( L(parameter set|response on that trial) )

    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
    #Define the variables that are important for the likelihood estimation process
        response = int(df.loc[trial , "Choice"]) - 1 #the stimulus shown on this trial
        
        if trial == 0: 
            PEs = np.array([0 , 0 , 0])
            LPs = np.array([0 , 0 , 0])
            Novs = np.array([0 , 0 , 0])
        else: 
            ##Wacht, maar ik gebruik normaal gezien de vorige trial om te schatten eh, jij ooeennnnnn 
            PEs = np.array([df.loc[trial - 1 , "PE_1"] , df.loc[trial - 1 , "PE_2"] , df.loc[trial - 1 , "PE_3"]])
            LPs = np.array([df.loc[trial - 1 , "LP_1"] , df.loc[trial - 1 , "LP_2"] , df.loc[trial - 1 , "LP_3"]])
            Novs = np.array([df.loc[trial - 1 , "Nov_1"] , df.loc[trial - 1 , "Nov_2"] , df.loc[trial - 1 , "Nov_3"]])
        
        loglikelihoods = np.log(softmax(parameter_set , PEs , LPs , Novs)) 

        # then select the probability of the actual response given the parameter set
        current_loglikelihood = loglikelihoods[response]

        # Add L(parameter set|current response) to the total log likelihood
        summed_logL = summed_logL + current_loglikelihood

    return summed_logL

#Define columns for output file
column_list = ["Generating_PE", "Generating_LP" , "Generating_Nov" , "Used_PE", "Used_LP" , "Used_Nov" , "AIC" , "generating_model" , "used_model"]
estimated_data = pd.DataFrame(columns=column_list)

PE_values = np.random.uniform(-1 , 0 , 1000)
np.random.shuffle(PE_values)
LP_values = np.random.uniform(0 , 1 , 1000)
np.random.shuffle(LP_values)
Nov_values = np.random.uniform(0 , 1 , 1000)
np.random.shuffle(Nov_values)

#parameter_values = np.column_stack([PE_values , LP_values , Nov_values])

"""
parameter_values = np.array([[1 , 0 , 0 ] , 
                            [0 , 1 , 0] , 
                            [0 , 0 , 1] , 
                            [1 , 1 , 0] , 
                            [0 , 1 , 1] , 
                            [1 , 1 , 1]])
"""

values_PE = np.random.uniform(-1 , 0 , 100) 
values_LP = np.random.uniform(0 , 1 , 100) 
values_Nov = np.random.uniform(0 , 1 , 100)

np.random.shuffle(values_PE)
np.random.shuffle(values_LP)
np.random.shuffle(values_Nov)

parameter_values = np.column_stack([values_PE , values_LP , values_Nov])

#Define number of simulations I want to do; here this is equivalent to the number of model recoveries I want
n_simulations = 100

for i in range(parameter_values.shape[0]): 
    
    weight_PE , weight_LP , weight_Nov = parameter_values[i , : ]
    
    for simulation in range(n_simulations): 
        
        #Generate a dataset first 
        os.chdir("/Users/wardclaeys/OneDrive - UGent/Model_recovery")
        generate_dataset(n = 150 , weight_PE = weight_PE , weight_LP = weight_LP, weight_Nov = weight_Nov)
        
        if __name__ == '__main__':
            #Extract path to data folder
            
            directory = "/Users/wardclaeys/OneDrive - UGent/Model_recovery"
        
            #Get the file I just generated (or at least a list of files, but it should be just the one)
            filelist = os.listdir(directory)
            filtered_filelist = [x for x in filelist if x[-3::]=="csv"]
            filtered_filelist = [x for x in filtered_filelist if x != "Model_recovery_results.csv"]
            filtered_filelist = [x for x in filtered_filelist if x[-13::] != "Simulated.csv"]
        
            #Go to that directory
            os.chdir(directory)
            
            idx = -1
            for file in filtered_filelist:
                idx +=1
        
                #Minimize negative loglikelihood
                df = pd.read_csv("Check_this_file.csv") 
                #optimization_output = optimize.minimize(fun = likelihood , x0 = [start_params, df]) #, options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})
                
                for recovery in range(parameter_values.shape[0]): 
                    
                    #Get the information in the file; the "Generating" variables are the variables used to generate the dataset 
                    #The "Used" variables are the ones used to get the LL 
                    estimated_data.loc[recovery + simulation * parameter_values.shape[0] + i * n_simulations , ["Generating_PE", "Generating_LP" , "Generating_Nov"]] = [weight_PE , weight_LP , weight_Nov]
                    estimated_data.loc[recovery + simulation * parameter_values.shape[0] + i * n_simulations , "generating_model"] = i + 1
                    
                    estimated_data.loc[recovery + simulation * parameter_values.shape[0] + i * n_simulations , ["Used_PE", "Used_LP" , "Used_Nov"]] = parameter_values[recovery , : ]
                    estimated_data.loc[recovery + simulation * parameter_values.shape[0] + i * n_simulations , "used_model"] = recovery + 1
                    
                    LL = likelihood(parameter_values[recovery , : ] , df)
                    
                    #Calculate the AIC; which is -2 * ln(Likelihood) + 2k
                    AIC = -2 * LL + 2 * parameter_values.shape[0]
                    
                    estimated_data.loc[recovery + simulation * parameter_values.shape[0] + i * n_simulations , "AIC"] = AIC
                    
                """
                #print("estimated learning rate is: {0} and estimated inverse temperature is: {1}.\n\n".format(lr, inv_temp))
                #Store everything in output file
                estimated_data.loc[i , ["Current_PE", "Current_LP" , "Current_Nov"]] = [df.loc[0 , "weight_PE"] , df.loc[0 , "weight_LP"] , df.loc[0 , "weight_Nov"]]
                #estimated_data.loc[idx , ["Estimated_PE", "Estimated_LP" , "Estimated_Nov", "Negative_LogL"]] = [estimated_parameters[0] , estimated_parameters[1] , estimated_parameters[2] , LL]
                estimated_data.loc[i , ["Estimated_PE" , "Estimated_LP" , "Estimated_Nov"]] = [estimated_parameters[0] , estimated_parameters[1] , estimated_parameters[2]]
                """
                    
            #Write results of parameter fitting
        os.chdir("/Users/wardclaeys/OneDrive - UGent/Model_recovery")
        estimated_data.to_csv("Model_recovery_results.csv", columns = column_list, float_format ='%.3f')
        print("End of fitting procedure")
        
        
        
        
        
        
        