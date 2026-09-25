#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:23:24 2026

@author: wardclaeys
"""



import numpy as np 
import pandas as pd 


"""
What do I need?
 
    - A current task 
    - A correct vs incorrect 
    - A boundary, but I want to loop over multiple ones to see if all can be recovered 
    - To keep as close as possible, the procedure should also stay in castle as long as it's correct, only do meta-trial if mistake (nice difference between meta-trial and trial)
    - Chosen and current location? So I can calculate a PE? Maybe this can be done differently though (see other script where you simulated data) 
    - Meta trial or not? 
    
"""

def sigmoid(x, slope , shift):
    return 1 / (1 + np.exp(-slope * x + shift * slope))

def stay_leave_decision(boundaries , current_task , RAs):

    if current_task == 1: 
        #Get the normalized value for the decision
        probability = sigmoid(RAs[0] , boundaries[2] , boundaries[0])
        
        #If the prob is high, then that's a sign to move on to a more difficult task, if not, we stay 
        #The first prob is the decrement one (which is not possible, since it's the easiest task), the second is the stay one and the third is the increment one         
        stay_leave = [0 , 1 - probability , probability]
        
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
        
    return stay_leave

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
    
    Increase = np.random.uniform(0 , 1 , 1)
    Decrease = np.random.uniform(0 , 1 , 1)
    slope = np.random.uniform(0 , 20 , 1)
    
    boundaries = [min(Increase) , min(Decrease) , min(slope)]
    #boundaries = [0.5 , 0.25 , 0.75 , 0.5] 
    RAs = [list(np.repeat(0 , 10)) , list(np.repeat(0 , 10)) , list(np.repeat(0 , 10))] 
    
    slope = np.random.uniform(0 , 20 , 1)
    
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
            
            #current_task = 1
        
        #And then we save that data to the file 
        data.loc[trial , ["Probability_correct_1" , "Probability_correct_2" , "Probability_correct_3"]] = task_probabilities
        
        data.loc[trial , "Current_task"] = current_task 
    
    
    data.loc[ : , ["Boundary_Increase" , "Boundary_Decrease" , "slope"]] = boundaries 
    
    return data
    
    #data.to_csv("Simulated_data.csv", columns = column_list, float_format ='%.3f')

