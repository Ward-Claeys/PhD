#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:31:36 2026

@author: wardclaeys
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


def generate_dataset(n = 200 , weight_PE = -0.5 , weight_LP = 0.5 , weight_Nov = 0.5): 
    ##Initialize the probabilities of selecting a task, just based on the probability to be correct 
    #We'll update these probabilities every time a task is selected
    initialize_probabilities = [0.5 , 0.25 , 0.125]
    
    data = pd.DataFrame()
    
    data.loc[0 , "Trial_nr"] = 0
    data.loc[0 , "Choice"] = 0
    data.loc[0 , "Reward"] = 0
    data.loc[0 , ["probability_1" , "probability_2" , "probability_3"]] = initialize_probabilities
    data.loc[0 , ["PE_1" , "PE_2" , "PE_3"]] = [0 , 0 , 0]
    data.loc[0 , ["LP_1" , "LP_2" , "LP_3"]] = [0 , 0 , 0]
    data.loc[0 , ["Nov_1" , "Nov_2" , "Nov_3"]] = [0 , 0 , 0]
    task_options = [1 , 2 , 3]
    
    PE_1 , PE_2 , PE_3 = 0 , 0 , 0
    LP_1 , LP_2 , LP_3 = 0 , 0 , 0
    novelty_1 , novelty_2 , novelty_3 = 99 , 99 , 99
    
    data.loc[ : , ["weight_PE" , "weight_LP" , "weight_Nov"]] = [weight_PE , weight_LP , weight_Nov]
    
    lambd = 0.03
    
    rolling_average_1 = []
    rolling_average_2 = []
    rolling_average_3 = []
    
    for i in range(n): 
        
        data.loc[i , "Trial_nr"] = i
        
        if i == 0: 
            task_choice = np.random.choice(task_options)
            
            data.loc[i , ["model_1_probability" , "model_2_probability" , "model_3_probability"]]= [0.33 , 0.33 , 0.33]
            
        else: 
            
            #Max seems to go to about 10 or something, so adjust so it's about between 0 and 1
            adjusted_novelty_1 = novelty_1 / 10
            adjusted_novelty_2 = novelty_2 / 10 
            adjusted_novelty_3 = novelty_3 / 10 
            
            numerator_1 = np.exp(weight_PE * PE_1 + weight_LP * LP_1 + weight_Nov * adjusted_novelty_1) 
            numerator_2 = np.exp(weight_PE * PE_2 + weight_LP * LP_2 + weight_Nov * adjusted_novelty_2) 
            numerator_3 = np.exp(weight_PE * PE_3 + weight_LP * LP_3 + weight_Nov * adjusted_novelty_3) 
        
            model_1 = numerator_1 / np.sum([numerator_1 , numerator_2 , numerator_3])
            model_2 = numerator_2 / np.sum([numerator_1 , numerator_2 , numerator_3])
            model_3 = numerator_3 / np.sum([numerator_1 , numerator_2 , numerator_3])
            
            task_choice = np.random.choice(task_options , p = [model_1 , model_2 , model_3])
            
            data.loc[i , ["model_1_probability" , "model_2_probability" , "model_3_probability"]]= [model_1 , model_2 , model_3]
            
        task_probabilities = list(data.loc[i , ["probability_1" , "probability_2" , "probability_3"]])
        
        data.loc[i , "Choice"] = task_choice
        
        if task_choice == 1: 
            prob_correct = data.loc[i , "probability_1"]
            correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
            
            rolling_average_1.append(correct)
            PE_1 = 1 - np.mean(rolling_average_1[-10 : ])
            
            #PE_1 = correct - data.loc[i , "probability_1"]
            
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
            
            novelty_1 = 0
            novelty_2 += 1
            novelty_3 += 1
    
        elif task_choice == 2: 
            prob_correct = data.loc[i , "probability_2"]
            correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
            
            rolling_average_2.append(correct)
            PE_2 = 1 - np.mean(rolling_average_2[-10 : ])
            
            #PE_2 = correct - data.loc[i , "probability_2"]
            
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
            
            novelty_2 = 0
            novelty_1 += 1 
            novelty_3 += 1
            
        else: 
            prob_correct = data.loc[i , "probability_3"]
            correct = np.random.choice([1 , 0] , p = [prob_correct , 1 - prob_correct])
            
            rolling_average_3.append(correct)
            PE_3 = 1 - np.mean(rolling_average_3[-10 : ])
            
            #PE_3 = correct - data.loc[i , "probability_3"]
            
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
            
            novelty_3 = 0 
            novelty_1 += 1
            novelty_2 += 1 
        
        if i != (n - 1): 
            data.loc[i + 1 , ["probability_1" , "probability_2" , "probability_3"]] = task_probabilities
            
        data.loc[i , ["PE_1" , "PE_2" , "PE_3"]] = [PE_1 , PE_2 , PE_3]
        data.loc[i , ["LP_1" , "LP_2" , "LP_3"]] = [LP_1 , LP_2 , LP_3]
        data.loc[i , ["Nov_1" , "Nov_2" , "Nov_3"]] = [novelty_1  , novelty_2  , novelty_3 ]
        data.loc[i , "Reward"] = correct 
        
        #Reset probabilities back to chance level every 50 trials 
        if i % 50 == 0: 
            data.loc[i + 1 , ["probability_1" , "probability_2" , "probability_3"]] = initialize_probabilities
    
    data.to_csv("Check_this_file.csv")
    
    return 






