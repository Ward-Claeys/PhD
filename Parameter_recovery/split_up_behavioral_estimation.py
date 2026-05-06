#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:56:26 2026

@author: wardclaeys
"""

# Import modules
import os, sys
import numpy as np
import pandas as pd
from scipy import optimize

# Avoid warnings
import warnings

warnings.filterwarnings("ignore")

os.chdir("/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery")

def softmax(values = np.array([0.5]) , PEs = np.array([0.5 , 0.5 , 0.5]) , LPs = np.array([0.5 , 0.5 , 0.5])  , Novs = np.array([0.5 , 0.5 , 0.5])):
    
    numerator_1 = np.exp(values[0] * PEs[0] + values[1] * LPs[0] + values[2] * Novs[0])
    numerator_2 = np.exp(values[0] * PEs[1] + values[1] * LPs[1] + values[2] * Novs[1])
    numerator_3 = np.exp(values[0] * PEs[2] + values[1] * LPs[2] + values[2] * Novs[2])
    
    denom = np.sum([numerator_1 , numerator_2 , numerator_3])
    
    response_probabilities = [numerator_1 / denom, numerator_2 / denom, numerator_3 / denom]
    
    #response_probabilities = np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs) / np.sum(np.exp(values[0][0] * PEs + values[0][1] * LPs + values[0][2] * Novs))
    
    return response_probabilities


def add_variables(data): 
    
    df = pd.read_csv(data)  # Read data
    
    df = df.query("Chosen_castle != -9999")
    df = df.query("Chosen_castle != 4")
    
    #Get the indices of the relevant trials => i.e., the meta trials
    #Note that the trial number start from 1 and we index from 0 in python
    indices = df.query("Accuracy == 0")["Trial_number"]
    indices = indices[ : -1]
    
    #df = df.iloc[indices]
    
    ntrials = df.shape[0]  # Extract number of trials

    novelty_1 , novelty_2 , novelty_3 = 99 , 99 , 99 
    
    for trial in range(ntrials):
        #I will go for the probability to be correct as the mean of the last 10 trials
        #LP as the change with the new trial in there 
        #nov just the same as elsewhere
        
    #Define the variables that are important for the likelihood estimation process
        response = int(df.loc[trial , "Chosen_castle"]) - 1 #the stimulus shown on this trial
        
        if trial == 0: 
            
            PE_1 , PE_2 , PE_3 = 0 , 0 , 0 
            LP_1 , LP_2 , LP_3 = 0 , 0 , 0 
            
            if response == 0: 
                novelty_1 = 0
            elif response == 1: 
                novelty_2 = 0 
            else: 
                novelty_3 = 0 
            
        else: 
            
            #Append the accuracy of the current trial in a specific condition
            if response == 0: 
                #moving_window_1.append(int(df.loc[trial , "Accuracy"]))
                #probability_1 = np.mean(moving_window_1[-10 : ])
                
                #PE_1 = int(df.loc[trial , "Accuracy"]) - probability_1 
                
                #Nothing changes with the chance tree for this castle 
                PE_1 = np.abs(df.loc[trial , "Correct_location"] - df.loc[trial , "Chosen_location"])
                
                novelty_1 = 0
                novelty_2 += 1
                novelty_3 += 1
                
                if trial != 1: 
                    LP_1 = df.loc[trial - 2 , "PE_1"] - df.loc[trial - 1 , "PE_1"]
                
            elif response == 1: 
                #moving_window_2.append(int(df.loc[trial , "Accuracy"]))
                #probability_2 = np.mean(moving_window_2[-10 : ])
                
                #PE_2 = int(df.loc[trial , "Accuracy"]) - probability_2
                
                #Here, it does change though
                #PE_2 = np.abs(df.loc[trial , "Correct_location"] - df.loc[trial , "Chosen_location"])
                
                PE_2 = 0 
                
                #If they got the wrong location, then it's an error. Go into the statement and then decide how bad the error is 
                if (df.loc[trial , "Correct_location"] != df.loc[trial , "Chosen_location"]):
                    #If it's a different side, then it's an error on the first "branch" 
                    #If it's not the same side, the statement is true and add 1 to the error term 
                    PE_2 += 1 * (1 * (df.loc[trial , "Correct_location"] < 2) != 1 * (df.loc[trial , "Chosen_location"] < 2))
                    
                    ##Then on the second branch it's an even-odd thing. 
                    #If same side, both even or both odd and then it's correct on the second branch. If not, add one for the error 
                    PE_2 += 1 * ((df.loc[trial , "Correct_location"] % 2) != (df.loc[trial , "Chosen_location"] % 2))
                
                novelty_2 = 0
                novelty_1 += 1 
                novelty_3 += 1  
                
                if trial != 1: 
                    LP_2 = df.loc[trial - 2 , "PE_2"] - df.loc[trial - 1 , "PE_2"]
                
            else: 
                #moving_window_3.append(int(df.loc[trial , "Accuracy"]))
                #probability_3 = np.mean(moving_window_3[-10 : ])
                
                #PE_3 = int(df.loc[trial , "Accuracy"]) - probability_3
                
                #PE_3 = np.abs(df.loc[trial , "Correct_location"] - df.loc[trial , "Chosen_location"])
                
                
                PE_3 = 0 
                
                #If they got the wrong location, then it's an error. Go into the statement and then decide how bad the error is 
                if (df.loc[trial , "Correct_location"] != df.loc[trial , "Chosen_location"]):
                    #If it's a different side, then it's an error on the first "branch" 
                    #If it's not the same side, the statement is true and add 1 to the error term 
                    PE_3 += 1 * (1 * (df.loc[trial , "Correct_location"] < 4) != 1 * (df.loc[trial , "Chosen_location"] < 4))
                    
                    #For the second branch; choosing the same side means going to either 0 , 1 , 4 , 5 OR going to 2 , 3 , 6 , 7 
                    #So if both chosen and correct are in set_1, then it's the same decision, so not an error on the second branch 
                    #If one is in the set and the other one not, then it's an error on the second branch 
                    set_1 = [2 , 3 , 6 , 7]
                    
                    PE_3 += 1 * ((df.loc[trial , "Correct_location"] in set_1) != (df.loc[trial , "Chosen_location"] in set_1))
                    
                    ##Then on the third branch it's an even-odd thing. 
                    #If same side, both even or both odd and then it's correct on the second branch. If not, add one for the error 
                    PE_3 += 1 * ((df.loc[trial , "Correct_location"] % 2) != (df.loc[trial , "Chosen_location"] % 2))
                
                novelty_3 = 0  
                novelty_2 += 1 
                novelty_1 += 1  
                
                if trial != 1: 
                    LP_3 = df.loc[trial - 2 , "PE_3"] - df.loc[trial - 1 , "PE_3"]
            
        df.loc[trial , ["PE_1" , "PE_2" , "PE_3"]] = [PE_1 , PE_2 , PE_3]
        df.loc[trial , ["LP_1" , "LP_2" , "LP_3"]] = [LP_1 , LP_2 , LP_3]
        df.loc[trial , ["Nov_1" , "Nov_2" , "Nov_3"]] = [novelty_1 , novelty_2 , novelty_3] 
    
    return df 

# Likelihood function for empirical data
#It only gets calculated for trials where participants do a meta-decision here, not for all trials 
def likelihood(parameter_set, data):

    df = pd.read_csv(data)  # Read data
    
    #Get the indices of the relevant trials => i.e., the meta trials
    #Note that the trial number start from 1 and we index from 0 in python
    indices = df.query("Accuracy == 0")["Trial_number"]
    indices = indices[ : -1]
    
    #df = df.iloc[indices]
    
    ntrials = df.shape[0]  # Extract number of trials

    # Start the likelihood estimation process: summed_logL = log(L(parameter set|data))
    # log(L(parameter set|data)) = sum( log( L(parameter set|response) ) for trial in trials)
    summed_logL = 0  # this is calculated by summing over trials the log( L(parameter set|response on that trial) )
    
    meta_trials = 0 
    # trial-loop: calculate log(L(parameter set|response)) on each trial
    for trial in range(ntrials):
        
        #Here +1 as trial is zero based indexing and indices is the trial number and so is one based indexing => small adjustment 
        if (trial + 1) in indices: 
            
            meta_trials += 1 
            estimated_data.loc[sub , "n_trials"] = meta_trials
            
            if trial != 0: 
                PEs = [df.loc[trial - 1 , "PE_1"] , df.loc[trial - 1  , "PE_2"] , df.loc[trial - 1  , "PE_3"]]
                LPs = [df.loc[trial - 1  , "LP_1"] , df.loc[trial - 1  , "LP_2"] , df.loc[trial - 1  , "LP_3"]]
                Novs = [df.loc[trial - 1  , "Nov_1"] , df.loc[trial - 1  , "Nov_2"] , df.loc[trial - 1  , "Nov_3"]]
            else: 
                PEs = [0 , 0 , 0]
                LPs = [0 , 0 , 0]
                Novs = [0 , 0 , 0]
            
            #Here I use PEs, LPs and Novs which are define in the beginning of the loop and contain the PEs, LPs and Novs of the previous trial 
            #Previous trial as this information is what is available to participants to make a decision. 
            loglikelihoods = np.log(softmax(parameter_set , PEs , LPs , Novs)) 
            
            response = int(df.loc[trial , "Chosen_castle"]) - 1 #the stimulus shown on this trial
            
            # then select the probability of the actual response given the parameter set
            current_loglikelihood = loglikelihoods[response]

            # Add L(parameter set|current response) to the total log likelihood
            summed_logL = summed_logL + current_loglikelihood
    
    return -summed_logL 

    return -summed_logL

#Define columns for output file
column_list = ["Estimated_PE", "Estimated_LP" , "Estimated_Nov", "Negative_LogL" , "sub_id" , "n_trials"]
estimated_data = pd.DataFrame(columns=column_list)

if __name__ == '__main__':
    #Extract path to data folder
    
    #directory = "/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery/Pilot_17th_March"
    directory = "/Users/wardclaeys/Documents/Github/PhD/experiment/Data_PhD"

    #Get a list of files in the folder and filter on csv files that are not previous fitting results
    filelist = os.listdir(directory)
    filtered_filelist = [x for x in filelist if x[-3::]=="csv"]
    filtered_filelist = [x for x in filtered_filelist if x != "Fitting_results_participants.csv"]
    filtered_filelist = [x for x in filtered_filelist if x[-19::] != "Fitting_results.csv"]

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
        
        ##Add the relevant variables to the data file and then save it. Here I overwrite the file as I don't really need it afterwards. 
        data = add_variables(file)
        data.to_csv("Fitting_results.csv")
        
        #Minimize negative loglikelihood
        optimization_output = optimize.minimize(fun = likelihood , x0 = start_params , args = "Fitting_results.csv" , options = {'maxfev':10000, 'xatol':0.00001, 'return_all':0})
        
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



