#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:55:10 2026

@author: wardclaeys
"""
import numpy as np 

def sigmoid(x, slope , shift):
    return 1 / (1 + np.exp(-slope * x + shift * slope))

def stay_leave_decision(boundaries , current_task , RAs):

    if current_task == 1: 
        #Get the normalized value for the decision
        
        probability = sigmoid(RAs[0] , boundaries[2] , boundaries[0])
        
        #If the prob is high, then that's a sign to move on to a more difficult task, if not, we stay 
        #The first prob is the decrement one (which is not possible, since it's the easiest task), the second is the stay one and the third is the increment one 
        
        stay_leave = [0 , 1 - probability , probability]
        #prob = RAs[0] - noise 
        #stay_leave = [0 , 1 - prob , prob]
        
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
        
        #prob = RAs[2] - noise
        #stay_leave = [0 , 1 - prob , prob]
        #print(probability)
        
    return stay_leave