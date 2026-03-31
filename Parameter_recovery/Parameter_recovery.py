#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:31:36 2026

@author: wardclaeys
"""

"""
Pr(castle i) = exp(γQ(𝑠𝑡 ,𝑎𝑡))
                ∑ exp(γQ(𝑠𝑡 ,𝑎𝑡))

Q(𝑠𝑡 , 𝑎𝑡) <- Q(𝑠𝑡 , 𝑎𝑡) + α(𝑟𝑡+1 - Q(𝑠𝑡 , 𝑎𝑡))
"""

"""
Dus effe recap van alles: 
    We gaan data gaan simuleren. Hoe gaan we dat doen? 
    We hebben elke keer kansen op selectie van een bepaalde taak equivalent aan kans op succes; in begin basically 0.5 - 0.25 - 0.125
    We selecteren een bepaalde taak obv een sigmoid functie en dan kiezen we die. 
    Elke keer we een taak doen, dan gaat de kans op succes naar boven. Kan via R-W model 
    En dan gaan we telkens zo door eigenlijk. 
    
    Voor selectie van de taak gebruiken we dan de gekozen parameters; PE, LP, Nov. 
    We kiezen bepaalde weights voor die parameters en gebruiken ze dan om te selecteren en dan gaan we verder. 
"""



import numpy as np 
import pandas as pd 

def softmax(x):

    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

##Initialize the probabilities of selecting a task, just based on the probability to be correct 
#We'll update these probabilities every time a task is selected
initialize_probabilities = [0.5 , 0.25 , 0.125]

data = pd.DataFrame()

data.loc[0 , "Choice"] = 0
data.loc[0 , ["probability_1" , "probability_2" , "probability_3"]] = initialize_probabilities
data.loc[0 , ["PE_1" , "PE_2" , "PE_3"]] = [0 , 0 , 0]
data.loc[0 , ["LP_1" , "LP_2" , "LP_3"]] = [0 , 0 , 0]
data.loc[0 , ["Nov_1" , "Nov_2" , "Nov_3"]] = [0 , 0 , 0]
task_options = [1 , 2 , 3]

PE_1 , PE_2 , PE_3 = 0 , 0 , 0
novelty_1 , novelty_2 , novelty_3 = 0 , 0 , 0

weight_PE = -0.5
weight_LP = 0.5 
weight_Nov = 0.5

lambd = 0.05

for i in range(100): 
    
    if i == 0: 
        task_choice = np.random.choice(task_options)
    else: 
        numerator_1 = np.exp(weight_PE * data.loc[i - 1 , "PE_1"] + weight_LP * data.loc[i - 1 , "LP_1"] + weight_Nov * data.loc[i - 1 , "Nov_1"]) 
        numerator_2 = np.exp(weight_PE * data.loc[i - 1 , "PE_2"] + weight_LP * data.loc[i - 1 , "LP_2"] + weight_Nov * data.loc[i - 1 , "Nov_2"]) 
        numerator_3 = np.exp(weight_PE * data.loc[i - 1 , "PE_3"] + weight_LP * data.loc[i - 1 , "LP_3"] + weight_Nov * data.loc[i - 1 , "Nov_3"]) 
    
        model_1 = numerator_1 / np.sum([numerator_1 , numerator_2 , numerator_3])
        model_2 = numerator_2 / np.sum([numerator_1 , numerator_2 , numerator_3])
        model_3 = numerator_3 / np.sum([numerator_1 , numerator_2 , numerator_3])
        
        task_choice = np.random.choice(task_options , p = [model_1 , model_2 , model_3])
        
    task_probabilities = list(data.loc[i , ["probability_1" , "probability_2" , "probability_3"]])
    
    data.loc[i , "Choice"] = task_choice
    
    if task_choice == 1: 
        prob_correct = data.loc[i , "probability_1"]
        correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
        
        PE_1 = correct - data.loc[i , "probability_1"]
        
        if i == 0: 
            PE_2 , PE_3 = 0 , 0
            LP_1 , LP_2 , LP_3 = 0 , 0 , 0
        else: 
            PE_2 = data.loc[i - 1 , "PE_2"]
            PE_3 = data.loc[i - 1 , "PE_3"]
            
            LP_1 = data.loc[i - 1 , "PE_1"] - PE_1
            LP_2 = data.loc[i - 1 , "PE_2"]
            LP_3 = data.loc[i - 1 , "PE_3"]
        
        #Update the probability of being correct by adding lambda * (1 - p(t - 1))
        task_probabilities[0] = data.loc[i , "probability_1"] + lambd * (1 - data.loc[i , "probability_1"])
        #The not chosen probabilities remain the same
        task_probabilities[1] = data.loc[i , "probability_2"]
        task_probabilities[2] = data.loc[i , "probability_3"]
        
        novelty_1 += 1
        novelty_2 = 0
        novelty_3 = 0

    elif task_choice == 2: 
        prob_correct = data.loc[i , "probability_2"]
        correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
        
        PE_2 = correct - data.loc[i , "probability_2"]
        
        if i == 0: 
            PE_1 , PE_3 = 0 , 0
            LP_1 , LP_2 , LP_3 = 0 , 0 , 0
        else: 
            PE_1 = data.loc[i - 1 , "PE_1"]
            PE_3 = data.loc[i - 1 , "PE_3"]
            
            LP_2 = data.loc[i - 1 , "PE_2"] - PE_2
            LP_1 = data.loc[i - 1 , "PE_1"]
            LP_3 = data.loc[i - 1 , "PE_3"]
        
        #Update the probability of being correct by adding lambda * (1 - p(t - 1))
        task_probabilities[1] = data.loc[i , "probability_2"] + lambd * (1 - data.loc[i , "probability_2"])
        #The not chosen probabilities remain the same
        task_probabilities[0] = data.loc[i , "probability_1"]
        task_probabilities[2] = data.loc[i , "probability_3"]
        
        novelty_2 += 1
        novelty_1 = 0
        novelty_3 = 0
        
    else: 
        prob_correct = data.loc[i , "probability_3"]
        correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
        
        PE_3 = correct - data.loc[i , "probability_3"]
        
        if i == 0: 
            PE_1 , PE_2 = 0 , 0
            LP_1 , LP_2 , LP_3 = 0 , 0 , 0
        else: 
            PE_1 = data.loc[i - 1 , "PE_1"]
            PE_2 = data.loc[i - 1 , "PE_2"]
            
            LP_3 = data.loc[i - 1 , "PE_3"] - PE_3
            LP_1 = data.loc[i - 1 , "PE_1"]
            LP_2 = data.loc[i - 1 , "PE_2"]
        
        #Update the probability of being correct by adding lambda * (1 - p(t - 1))
        task_probabilities[2] = data.loc[i , "probability_3"] + lambd * (1 - data.loc[i , "probability_3"])
        #The not chosen probabilities remain the same
        task_probabilities[0] = data.loc[i , "probability_1"]
        task_probabilities[1] = data.loc[i , "probability_2"]
        
        novelty_3 += 1
        novelty_1 = 0
        novelty_2 = 0
    
    data.loc[i + 1 , ["probability_1" , "probability_2" , "probability_3"]] = task_probabilities
    data.loc[i , ["PE_1" , "PE_2" , "PE_3"]] = [PE_1 , PE_2 , PE_3]
    data.loc[i , ["LP_1" , "LP_2" , "LP_3"]] = [LP_1 , LP_2 , LP_3]
    data.loc[i , ["Nov_1" , "Nov_2" , "Nov_3"]] = [novelty_1 , novelty_2 , novelty_3]
    

