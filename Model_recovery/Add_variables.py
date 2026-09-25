#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:35:17 2026

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

os.chdir("/Users/wardclaeys/Documents/Github/PhD/Parameter_recovery/Model_recovery")

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
    moving_window_1 , moving_window_2 , moving_window_3 = list(np.repeat(0 , 10)) , list(np.repeat(0 , 10)) , list(np.repeat(0 , 10)) 
    
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
                
                moving_window_1.append(int(df.loc[trial , "Accuracy"]))
                probability_1 = np.mean(moving_window_1[-10 : ])
                probability_2 = np.mean(moving_window_2[-10 : ])
                probability_3 = np.mean(moving_window_3[-10 : ])
                
            elif response == 1: 
                novelty_2 = 0 
                
                moving_window_2.append(int(df.loc[trial , "Accuracy"]))
                probability_2 = np.mean(moving_window_2[-10 : ])
                probability_1 = np.mean(moving_window_1[-10 : ])
                probability_3 = np.mean(moving_window_3[-10 : ])
                
            else: 
                novelty_3 = 0 
                
                moving_window_3.append(int(df.loc[trial , "Accuracy"]))
                probability_3 = np.mean(moving_window_3[-10 : ])
                probability_2 = np.mean(moving_window_2[-10 : ])
                probability_1 = np.mean(moving_window_1[-10 : ])
            
        else: 
            
            #Append the accuracy of the current trial in a specific condition
            if response == 0: 
                moving_window_1.append(int(df.loc[trial , "Accuracy"]))
                probability_1 = np.mean(moving_window_1[-10 : ])
                
                #df.loc[trial , "accuracy_1"] = probability_1 
                
                #PE_1 = int(df.loc[trial , "Accuracy"]) - probability_1 
                
                #Nothing changes with the chance tree for this castle 
                PE_1 = np.abs(df.loc[trial , "Correct_location"] - df.loc[trial , "Chosen_location"])
                
                novelty_1 = 0
                novelty_2 += 1
                novelty_3 += 1
                
                if trial != 1: 
                    LP_1 = df.loc[trial - 2 , "PE_1"] - df.loc[trial - 1 , "PE_1"]
                
            elif response == 1: 
                moving_window_2.append(int(df.loc[trial , "Accuracy"]))
                probability_2 = np.mean(moving_window_2[-10 : ])
                
                #df.loc[trial , "accuracy_2"] = probability_2 
                
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
                moving_window_3.append(int(df.loc[trial , "Accuracy"]))
                probability_3 = np.mean(moving_window_3[-10 : ])
                
                #df.loc[trial , "accuracy_3"] = probability_3 
                
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
        df.loc[trial , ["accuracy_1" , "accuracy_2" , "accuracy_3"]] = [probability_1 , probability_2 , probability_3]
    
    return df 
