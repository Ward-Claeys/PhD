#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:50:23 2026

@author: wardclaeys

Model simulation 
"""


import numpy as np 
import pandas as pd 
from stay_leave_decision import stay_leave_decision


"""
What do I need?
 
    - A current task 
    - A correct vs incorrect 
    - A boundary, but I want to loop over multiple ones to see if all can be recovered 
    - To keep as close as possible, the procedure should also stay in castle as long as it's correct, only do meta-trial if mistake (nice difference between meta-trial and trial)
    - Chosen and current location? So I can calculate a PE? Maybe this can be done differently though (see other script where you simulated data) 
    - Meta trial or not? 
    
"""

def generate_dataset(n_trials = 100): 
    
    column_list = ["Current_task" , "Accuracy" , "Boundary_Task_1" , "Boundary_Task_2_low" , "Boundary_Task_2_high" , "Boundary_Task_3" , 
                   "Meta_trial" , "Probability_correct_1" , "Probability_correct_2" , "Probability_correct_3" , "Decision" , "Prob_Dec" , "Prob_Stay" , "Prob_Inc" , 
                   "accuracy_1" , "accuracy_2" , "accuracy_3" , "slope"]
    
    data = pd.DataFrame(columns = column_list)
    
    tasks = [1 , 2 , 3]
    
    #Update the probability of being correct by adding lambda * (1 - p(t - 1))
    #Probability on a task = (probability of trial before) + lambda * ( 1 - (probability of the trial before)) 
    
    #Initialize some variables 
    task_probabilities = [0.5 , 0.25 , 0.125]
    
    task_1 = np.random.uniform(0 , 1 , 1)
    task_2 = np.random.uniform(0 , 1 , 2)
    task_3 = np.random.uniform(0 , 1 , 1)
    slope = np.random.uniform(0 , 20 , 1)
    
    boundaries = [min(task_1) , max(task_2) , min(task_2) , min(task_3) , min(slope)]
    #boundaries = [0.5 , 0.25 , 0.75 , 0.5] 
    RAs = [list(np.repeat(0 , 10)) , list(np.repeat(0 , 10)) , list(np.repeat(0 , 10))] 
    
    slope = np.random.uniform(0 , 20 , 1)
    
    PE_1 , PE_2 , PE_3 , LP_1 , LP_2 , LP_3 = 0 , 0 , 0 , 0 , 0 , 0
    Nov_1 , Nov_2 , Nov_3 = 99 , 99 , 99 
    
    lambd = 0.05 
    
    for trial in range(n_trials): 
        
        #First trial is different because you don't have earlier information yet 
        if trial == 0: 
            #As there is no earlier info, the first decision is purely random 
            current_task = np.random.choice(tasks) 
            
            #Being correct is dependent on how good you are at a certain task 
            correct = np.random.choice([1 , 0] , p = [task_probabilities[current_task - 1] , 1 - task_probabilities[current_task - 1]]) 
            
            #Put accuracy in the file 
            data.loc[trial , "Accuracy"] = correct 
            
            #Add the accuracy to the rolling average of that task 
            RAs[current_task - 1].append(correct)
            
            #When you do a task, you become slightly better. The "slightly" here is scaled by a labmda paramter 
            task_probabilities[current_task - 1] = task_probabilities[current_task - 1] + lambd * (1 - task_probabilities[current_task - 1])
            #And then we save that data to the file 
            data.loc[trial , ["Probability_correct_1" , "Probability_correct_2" , "Probability_correct_3"]] = task_probabilities
            
        else: 
            
            previous_task = data.loc[trial - 1 , "Current_task"]
            
            RAs_values = [np.mean(RAs[0][-10 : ]) , np.mean(RAs[1][-10 : ]) , np.mean(RAs[2][-10 : ])]
            
            #Then I decide, based on the information whether to decrease, stay or increase 
            decision_probabilities = stay_leave_decision(boundaries , previous_task , RAs_values)
            #decision_probabilities = list((decision_probabilities[0][0] , decision_probabilities[1][0] , decision_probabilities[2][0]))
            #Select a decision then based on the probabilities 
            decision = np.random.choice([-1 , 0 , 1] , p = decision_probabilities) 
            data.loc[trial - 1 , "Decision"] = decision 
            
            current_task = previous_task + decision 
            
            #Being correct is dependent on how good you are at a certain task 
            correct = np.random.choice([1 , 0] , p = [task_probabilities[previous_task - 1] , 1 - task_probabilities[previous_task - 1]]) 
            #Put accuracy in the file 
            data.loc[trial , "Accuracy"] = correct 
            #Add the accuracy to the rolling average of that task 
            RAs[previous_task - 1].append(correct)
            #When you do a task, you become slightly better. The "slightly" here is scaled by a labmda parameter 
            task_probabilities[previous_task - 1] = task_probabilities[previous_task - 1] + lambd * (1 - task_probabilities[previous_task - 1])
            
            data.loc[trial - 1 , ["Prob_Dec" , "Prob_Stay" , "Prob_Inc"]] = decision_probabilities
            data.loc[trial - 1 , ["accuracy_1" , "accuracy_2" , "accuracy_3"]] = RAs_values
            
            if current_task == 1: 
                PE_1 = 1 - np.mean(RAs_values[0]) 
                PE_2 = data.loc[trial - 1 , "PE_2"]
                PE_3 = data.loc[trial - 1 , "PE_3"]
                
                LP_1 = data.loc[trial - 1 , "PE_1"] - PE_1 
                LP_2 = data.loc[trial - 1 , "LP_2"]
                LP_3 = data.loc[trial - 1 , "LP_3"] 
                
                Nov_1 = 0 
                Nov_2 += 1
                Nov_3 += 1 
                
            elif current_task == 2: 
                PE_1 = data.loc[trial - 1 , "PE_1"]
                PE_2 = 1 - np.mean(RAs_values[1])
                PE_3 = data.loc[trial - 1 , "PE_3"]
                
                LP_1 = data.loc[trial - 1 , "LP_1"]
                LP_2 = data.loc[trial - 1 , "PE_2"] - PE_2 
                LP_3 = data.loc[trial - 1 , "LP_3"]
                
                Nov_1 += 1 
                Nov_2 = 0
                Nov_3 += 1 
                
            else: 
                PE_1 = data.loc[trial - 1 , "PE_1"]
                PE_2 = data.loc[trial - 1 , "PE_2"]
                PE_3 = 1 - np.mean(RAs_values[2]) 
                
                LP_1 = data.loc[trial - 1 , "LP_1"]
                LP_2 = data.loc[trial - 1 , "PE_2"] 
                LP_3 = data.loc[trial - 1 , "LP_3"] - PE_3 
                
                Nov_1 += 1 
                Nov_2 += 1
                Nov_3 = 0
            
            #current_task = 1
        
        #And then we save that data to the file 
        data.loc[trial , ["Probability_correct_1" , "Probability_correct_2" , "Probability_correct_3"]] = task_probabilities
        
        data.loc[trial , "Current_task"] = current_task 
        data.loc[trial , ["PE_1" , "PE_2" , "PE_3" , "LP_1" , "LP_2" , "LP_3" , "Nov_1" , "Nov_2" , "Nov_3"]] = PE_1 , PE_2 , PE_3 , LP_1 , LP_2 , LP_3 , Nov_1 , Nov_2 , Nov_3
    
    
    data.loc[ : , ["Boundary_Task_1" , "Boundary_Task_2_high" , "Boundary_Task_2_low" , "Boundary_Task_3" , "slope"]] = boundaries 
    
    return data
    
    #data.to_csv("Simulated_data.csv", columns = column_list, float_format ='%.3f')






