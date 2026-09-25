#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:32:24 2026

@author: wardclaeys


Model recovery file 
"""


#import numpy as np 
#from stay_leave_decision import stay_leave_decision
from Model_simulation import generate_dataset as dataset_4_boundaries
from Model_simulation_Shared_boundaries import generate_dataset as dataset_shared_boundaries 
from Model_simulation_Independent_Selection import generate_dataset as dataset_independent_selection

import pandas as pd 
import numpy as np

recovery = pd.DataFrame()

current_model = list(np.repeat(0 , 10)) + list(np.repeat(1 , 10)) + list(np.repeat(2 , 10))
recovery.loc[ : , "Current_model"] = current_model 

for i in range(len(current_model)): 
    
    if current_model == 0: 
        data = dataset_4_boundaries(n_trials = 100)
    elif current_model == 1: 
        data = dataset_shared_boundaries(n_trials = 100)
    else: 
        data = dataset_independent_selection(n = 100 , weight_PE = 0 , weight_LP = 0 , weight_Nov = 0)
    
    
    
    

data = dataset_4_boundaries(100)
data = dataset_shared_boundaries(100)
data = dataset_independent_selection(100 , weight_PE = 0 , weight_LP = 0  , weight_Nov = 0 )


print(data)













