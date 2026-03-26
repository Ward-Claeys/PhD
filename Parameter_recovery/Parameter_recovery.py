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

def softmax(x):

    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities


for i in range(10): 
    
    data.loc[i , ["probability_1" , "probability_2" , "probability_3"]] = [0 , 0 , 0]
    
    task_probabilities = list(data.loc[i , ["probability_1" , "probability_2" , "probability_3"]])
    
    task_probabilities = softmax(task_probabilities)
    
    task_choice = np.random.choice(task_options , p = task_probabilities)
    
    if task_choice == 1: 
        prob_correct = data.loc[i , "probability_1"]
        correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
        
        PE_1 = correct - data.loc[i , "probability_1"]
    
    print("**********")
    print(i)
    print(task_probabilities)
    print(task_choice)
    
    
"""
Q(S,A) ← Q(S,A) + α (R + γ maxQ(S′,a) − Q(S,A))

R as correct or incorrect, so 1 or 0 => dependent on accuracy 
y is a discount parameter => fix
a learning rate => fix
"""



























