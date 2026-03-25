#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:04:31 2026

@author: wardclaeys
"""

import numpy as np
import matplotlib.pyplot as plt 
import os 

os.getcwd()

##Direct yourself to the folder with the data in first
os.chdir("/Users/wardclaeys/OneDrive - UGent/Modelling/data/Trouble_Shooting_Parameter_search")

##Read in the data 
#I'm still trying to figure out how to do this more efficiently, but can't seem to wrap my head around it...
PE01_LP01_NOV01_acc_1 = np.load("PE_-0.10_LP_0.10_Nov_0.10/model_accuracy_1.npy")
PE01_LP01_NOV01_acc_2 = np.load("PE_-0.10_LP_0.10_Nov_0.10/model_accuracy_2.npy")
PE01_LP01_NOV01_acc_3 = np.load("PE_-0.10_LP_0.10_Nov_0.10/model_accuracy_3.npy")

PE01_LP01_NOV06_acc_1 = np.load("PE_-0.10_LP_0.10_Nov_0.60/model_accuracy_1.npy")
PE01_LP01_NOV06_acc_2 = np.load("PE_-0.10_LP_0.10_Nov_0.60/model_accuracy_2.npy")
PE01_LP01_NOV06_acc_3 = np.load("PE_-0.10_LP_0.10_Nov_0.60/model_accuracy_3.npy")

PE01_LP06_NOV01_acc_1 = np.load("PE_-0.10_LP_0.60_Nov_0.10/model_accuracy_1.npy")
PE01_LP06_NOV01_acc_2 = np.load("PE_-0.10_LP_0.60_Nov_0.10/model_accuracy_2.npy")
PE01_LP06_NOV01_acc_3 = np.load("PE_-0.10_LP_0.60_Nov_0.10/model_accuracy_3.npy")

PE06_LP01_NOV01_acc_1 = np.load("PE_-0.60_LP_0.10_Nov_0.10/model_accuracy_1.npy")
PE06_LP01_NOV01_acc_2 = np.load("PE_-0.60_LP_0.10_Nov_0.10/model_accuracy_2.npy")
PE06_LP01_NOV01_acc_3 = np.load("PE_-0.60_LP_0.10_Nov_0.10/model_accuracy_3.npy")

PE06_LP06_NOV01_acc_1 = np.load("PE_-0.60_LP_0.60_Nov_0.10/model_accuracy_1.npy")
PE06_LP06_NOV01_acc_2 = np.load("PE_-0.60_LP_0.60_Nov_0.10/model_accuracy_2.npy")
PE06_LP06_NOV01_acc_3 = np.load("PE_-0.60_LP_0.60_Nov_0.10/model_accuracy_3.npy")

PE06_LP01_NOV06_acc_1 = np.load("PE_-0.60_LP_0.10_Nov_0.60/model_accuracy_1.npy")
PE06_LP01_NOV06_acc_2 = np.load("PE_-0.60_LP_0.10_Nov_0.60/model_accuracy_2.npy")
PE06_LP01_NOV06_acc_3 = np.load("PE_-0.60_LP_0.10_Nov_0.60/model_accuracy_3.npy")

PE01_LP06_NOV06_acc_1 = np.load("PE_-0.10_LP_0.60_Nov_0.60/model_accuracy_1.npy")
PE01_LP06_NOV06_acc_2 = np.load("PE_-0.10_LP_0.60_Nov_0.60/model_accuracy_2.npy")
PE01_LP06_NOV06_acc_3 = np.load("PE_-0.10_LP_0.60_Nov_0.60/model_accuracy_3.npy")

PE06_LP06_NOV06_acc_1 = np.load("PE_-0.60_LP_0.60_Nov_0.60/model_accuracy_1.npy")
PE06_LP06_NOV06_acc_2 = np.load("PE_-0.60_LP_0.60_Nov_0.60/model_accuracy_2.npy")
PE06_LP06_NOV06_acc_3 = np.load("PE_-0.60_LP_0.60_Nov_0.60/model_accuracy_3.npy")

##Put all files in a list to loop through it easier after this 
files = [PE01_LP01_NOV01_acc_1 , PE01_LP01_NOV01_acc_2 , PE01_LP01_NOV01_acc_3 , 
         PE01_LP01_NOV06_acc_1 , PE01_LP01_NOV06_acc_2 , PE01_LP01_NOV06_acc_3 , 
         PE01_LP06_NOV01_acc_1 , PE01_LP06_NOV01_acc_2 , PE01_LP06_NOV01_acc_3 , 
         PE06_LP01_NOV01_acc_1 , PE06_LP01_NOV01_acc_2 , PE06_LP01_NOV01_acc_3 , 
         PE06_LP06_NOV01_acc_1 , PE06_LP06_NOV01_acc_2 , PE06_LP06_NOV01_acc_3 , 
         PE06_LP01_NOV06_acc_1 , PE06_LP01_NOV06_acc_2 , PE06_LP01_NOV06_acc_3 , 
         PE01_LP06_NOV06_acc_1 , PE01_LP06_NOV06_acc_2 , PE01_LP06_NOV06_acc_3 , 
         PE06_LP06_NOV06_acc_1 , PE06_LP06_NOV06_acc_2 , PE06_LP06_NOV06_acc_3]

##Make an array to store the values of the last 100 accuracies in per model and per simulation 
mean_simulation = np.zeros((10 , 3 , 8))
##8 conditions and 3 files per condition
mean_accuracy = np.zeros((8 , 3))

file_nr = 0
simulation_nr = 0

for file in files: 
    ##Move down one for each time 3 files are read (as this is a new condition)
    #This is because we have 3 models per condition; we do three, then move down
    current_condition = file_nr % 3
    
    file_nr += 1
    
    #So if current condition is divisible by 3, we did all models for that condition, so we move one down in the array
    simulation_nr += (current_condition == 0)
    
    for simulation in range(file.shape[0]): 
        
        simulation_nr - 1 
        
        #Make a list of the current simulation to easily search for the last relevant index 
        s = list(file[simulation])
        
        #Get the index with the last value that is filled out (file was just filled with -9999's as it can't be predicted how many times a model was chosen)
        last_relevant_index = s.index(-9999)
        
        mean_simulation[simulation , current_condition , simulation_nr - 1] = np.mean(file[simulation][last_relevant_index - 100 : last_relevant_index])
    
    current_one = mean_simulation[: , current_condition , simulation_nr - 1]
    
    mean_accuracy[simulation_nr - 1 , current_condition] = np.mean(current_one[~ np.isnan(current_one)])





