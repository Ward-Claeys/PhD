#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:07:41 2025

@author: wardclaeys
"""

"""
Trial selection in here. Trial per trial training of the model, random selection of trials. 
All models seem to converge just fine (here plotted with both accuracy and loss function). 

Overall, the benefit of pretraining easier trials seems evident given tasks are dependent (i.e., tasks from easier models can be used in later models). 
Sometimes so extreme that model itself cannot learn task but model with pretraining on easier (i.e., composite task) can. 
"""

#%%
import numpy as np
from itertools import product

input_options = np.asarray(list(product([0, 1], repeat = 6)))

##Construct input matrix, will be used for all models
input_matrix = np.zeros((64 , 6))

input_matrix[ : , 0 : 6] = input_options

##Construct output matrix for model 1 (I know not efficient, but just wanted to check if it worked). 
output_matrix = np.zeros((64 , 2))

for i in range(output_matrix.shape[0]): 
    if ((input_matrix[i , 0] == 0) and (input_matrix[i , 1] == 1)) or ((input_matrix[i , 0] == 1) and (input_matrix[i , 1] == 0)): 
        output_matrix[i , 0] = 1

#If left is 1, right is 0 and reversed
output_matrix[ : , 1] = 1 - output_matrix[: , 0]


#%% Define the step we take in weight space

def step(model, X, y):
    """keep track of our gradients"""
    with tf.GradientTape() as tape:
     # make a prediction using the model and then calculate the loss
#        dict_X = array_to_dict(X)
        pred = model(inputs = X) 
        loss = categorical_crossentropy(y, pred)
	    # calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

#%%Basics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from itertools import product

n_epochs    = 100
n_trials    = 64 #this is the number of trials during an epoch, to keep it similar, go like this. 
n_simulations  = 10

#%% Model 1
history_1 = np.zeros((n_simulations , n_epochs))
accuracy_1 = np.zeros((n_simulations , n_epochs))

#Function for building model 1 
def build_model(): 
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(2 , activation = "softmax")
         ])
    model.build()
    
    return model

for sim in range(n_simulations): 
    #Build it
    model = build_model()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    for epoch in range(n_epochs):
        
        for trial in range(n_trials):
            
            # do a random shuffle of the stimuli (+ outputs)
            ix = np.random.permutation(range(input_matrix.shape[0]))
            
            x_shuffle = input_matrix[ix]
            t_shuffle = output_matrix[ix]
            
            index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
            
            current_trial = x_shuffle[index_trial] 
            current_output = t_shuffle[index_trial]
            
            # take a step in weight space
            step(model, current_trial, current_output)
            # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
        (history_1[sim , epoch] , accuracy_1[sim , epoch]) = model.evaluate(x_shuffle , t_shuffle)
        
##Get the predictions of the model (for model 2, the sequential one)
predictions_model = model.predict(input_matrix)
##predictions_model = model(input_matrix) 

mean_1, lower_1, upper_1 = [],[],[]
mean_6, lower_6, upper_6 = [],[],[]

ci = 0.95

for i in range (history_1.shape[1]):

    a = history_1[ : , i]

    MEAN = np.mean(a)
    mean_1.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_1.append(Lower)
    upper_1.append(Upper)
    
    a = accuracy_1[ : , i]

    MEAN = np.mean(a)
    mean_6.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_6.append(Lower)
    upper_6.append(Upper)

"""
plt.plot(mean_1 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_1))), upper_1, lower_1, color="green", alpha=0.25)

plt.plot(mean_6 , "-b" , label = "mean" , color = "red")
plt.fill_between(list(range(len(mean_6))), upper_6, lower_6, color="red", alpha=0.25)

plt.show()
"""
#%%Model 2

history_2_itself_matrix = np.zeros((n_simulations , n_epochs))
accuracy_2_itself_matrix = np.zeros((n_simulations , n_epochs))

##Construct the output matrix for model 2 (4 outputs). Again not efficient, but was again just to check it worked. 
output_matrix_2 = np.zeros((64 , 4))

for i in range(output_matrix_2.shape[0]): 
    if (input_matrix[i , 0] == 0 and input_matrix[i , 1] == 1) or (input_matrix[i , 0] == 1 and input_matrix[i , 1] == 0):
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            output_matrix_2[i , : ] = [1 , 0 , 0 , 0]
        else: 
            output_matrix_2[i , : ] = [0, 1 , 0 , 0]
    else: 
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            output_matrix_2[i , : ] = [0 , 0 , 1 , 0]
        else: 
            output_matrix_2[i , : ] = [0 , 0 , 0 , 1]

#Build model 2 that trains by itself, so 5 inputs to 4 outputs with a hidden layer in between (not sure if this should be added, but otherwise the XOR is impossible, so just left it in for now)
def model_2_itself(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    
    return model

for sim in range(n_simulations): 
    #Build it
    model_2 = model_2_itself()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model_2.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    for epoch in range(n_epochs):
        
        for trial in range(n_trials):
            # do a random shuffle of the stimuli (+ outputs)
            ix = np.random.permutation(range(input_matrix.shape[0]))
            x_shuffle = input_matrix[ix]
            t_shuffle = output_matrix_2[ix]
            
            index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
            
            current_trial = x_shuffle[index_trial] 
            current_output = t_shuffle[index_trial]
            
            # take a step in weight space
            step(model_2, current_trial, current_output)
            # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
        (history_2_itself_matrix[sim , epoch] , accuracy_2_itself_matrix[sim , epoch]) = model_2.evaluate(x_shuffle , t_shuffle)

mean_2, lower_2, upper_2 = [],[],[]
mean_7, lower_7, upper_7 = [],[],[]

ci = 0.95

for i in range (history_2_itself_matrix.shape[1]):

    a = history_2_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_2.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_2.append(Lower)
    upper_2.append(Upper)
    
    a = accuracy_2_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_7.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_7.append(Lower)
    upper_7.append(Upper)
    
"""
plt.plot(mean_2 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_2))), upper_2, lower_2, color="green", alpha=0.25)
plt.show()
"""
#%% Model 2 sequential

history_2_sequential_matrix = np.zeros((n_simulations , n_epochs))
accuracy_2_sequential_matrix = np.zeros((n_simulations , n_epochs))

#Then build model 2 for sequential learning. Here the input matrix is different with an extra 2 units, based on model 1
#Note that here the hidden layer is gone, and it's just the 7 (5 + 2) inputs on 4 outputs. Hidden layer is so to say hidden-hidden in the 2 new input units
def model_2_sequential(): 
    
    model = keras.Sequential([
        Input(shape = (8 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(4 , activation = "softmax")
         ])
    model.build()
    
    return model

#Make the new input unit with the predictions of the model 
input_matrix_2 = np.column_stack([input_matrix , predictions_model])

for sim in range(n_simulations): 
    #Build it
    model_2 = model_2_sequential()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model_2.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    for epoch in range(n_epochs):
        
        for trial in range(n_trials):
            # do a random shuffle of the stimuli (+ outputs)
            ix = np.random.permutation(range(input_matrix_2.shape[0]))
            x_shuffle = input_matrix_2[ix]
            t_shuffle = output_matrix_2[ix]
            
            index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
            
            current_trial = x_shuffle[index_trial] 
            current_output = t_shuffle[index_trial]
            
            # take a step in weight space
            step(model_2, current_trial, current_output)
            # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
        (history_2_sequential_matrix[sim , epoch] , accuracy_2_sequential_matrix[sim , epoch]) = model_2.evaluate(x_shuffle , t_shuffle)

model_2_predictions = model_2.predict(input_matrix_2)

mean_3, lower_3, upper_3 = [],[],[]
mean_8, lower_8, upper_8 = [],[],[]

ci = 0.95

for i in range (history_2_sequential_matrix.shape[1]):

    a = history_2_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_3.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_3.append(Lower)
    upper_3.append(Upper)
    
    a = accuracy_2_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_8.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_8.append(Lower)
    upper_8.append(Upper)

"""
plt.plot(mean_3 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_3))), upper_3, lower_3, color="green", alpha=0.25)
plt.show()
"""
#%%Model 3 sequential

##########################
## Make output matrix 3 ##
##########################

output_matrix_3 = np.zeros((64 , 8))

for i in range(output_matrix_3.shape[0]): 
    if (input_matrix[i , 0] == 0 and input_matrix[i , 1] == 1) or (input_matrix[i , 0] == 1 and input_matrix[i , 1] == 0):
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0): 
                output_matrix_3[i , : ] = [1 , 0 , 0 , 0 , 0 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 1 , 0 , 0 , 0 , 0 , 0 , 0]
        else: 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0):
                output_matrix_3[i , : ] = [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0]

    else: 
        if (input_matrix[i , 2] == 0 and input_matrix[i , 3] == 1) or (input_matrix[i , 2] == 1 and input_matrix[i , 3] == 0): 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0): 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]
        else: 
            if (input_matrix[i , 4] == 0 and input_matrix[i , 5] == 1) or (input_matrix[i , 4] == 1 and input_matrix[i , 5] == 0):
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0]
            else: 
                output_matrix_3[i , : ] = [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1]

history_3_sequential_matrix = np.zeros((n_simulations , n_epochs))
accuracy_3_sequential_matrix = np.zeros((n_simulations , n_epochs))

def model_3_sequential(): 
    
    model = keras.Sequential([
        Input(shape = (10 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    
    return model

input_matrix_3 = np.column_stack([input_matrix , model_2_predictions])

for sim in range(n_simulations): 
    #Build it
    model_3 = model_3_sequential()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model_3.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    for epoch in range(n_epochs):
        
        for trial in range(n_trials):
            # do a random shuffle of the stimuli (+ outputs)
            ix = np.random.permutation(range(input_matrix_3.shape[0]))
            x_shuffle = input_matrix_3[ix]
            t_shuffle = output_matrix_3[ix]
            
            index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
            
            current_trial = x_shuffle[index_trial] 
            current_output = t_shuffle[index_trial]
            
            # take a step in weight space
            step(model_3, current_trial, current_output)
            # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
        (history_3_sequential_matrix[sim , epoch] , accuracy_3_sequential_matrix[sim , epoch]) = model_3.evaluate(x_shuffle , t_shuffle)

mean_4, lower_4, upper_4 = [],[],[]
mean_9, lower_9, upper_9 = [],[],[]

ci = 0.95

for i in range (history_3_sequential_matrix.shape[1]):

    a = history_3_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_4.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_4.append(Lower)
    upper_4.append(Upper)
    
    a = accuracy_3_sequential_matrix[ : , i]

    MEAN = np.mean(a)
    mean_9.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_9.append(Lower)
    upper_9.append(Upper)
    
"""
plt.plot(mean_4 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_4))), upper_4, lower_4, color="green", alpha=0.25)
plt.show()
"""
#%% Model 3 by itself

history_3_itself_matrix = np.zeros((n_simulations , n_epochs))
accuracy_3_itself_matrix = np.zeros((n_simulations , n_epochs))

#Build model 2 that trains by itself, so 5 inputs to 4 outputs with a hidden layer in between (not sure if this should be added, but otherwise the XOR is impossible, so just left it in for now)
def model_3_itself(): 
    
    model = keras.Sequential([
        Input(shape = (6 ,)) , 
        Dense(3 , activation = "sigmoid") , 
        Dense(8 , activation = "softmax")
         ])
    model.build()
    
    return model

for sim in range(n_simulations): 
    #Build it
    model_3 = model_3_itself()
    
    loss_CE = keras.losses.CategoricalCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = 0.1)
    model_3.compile(optimizer = opt, 
                  loss = loss_CE , 
                  metrics=[keras.metrics.CategoricalAccuracy()])
    
    for epoch in range(n_epochs):
        
        for trial in range(n_trials):
            # do a random shuffle of the stimuli (+ outputs)
            ix = np.random.permutation(range(input_matrix.shape[0]))
            x_shuffle = input_matrix[ix]
            t_shuffle = output_matrix_3[ix]
            
            index_trial = np.random.randint(low = 0 , high = 63 , size = 1)
            
            current_trial = x_shuffle[index_trial] 
            current_output = t_shuffle[index_trial]
            
            # take a step in weight space
            step(model_3, current_trial, current_output)
            # to store intermediate results: bcs we don't use model.fit(), must explicitly calculate error for plotting
        (history_3_itself_matrix[sim , epoch] , accuracy_3_itself_matrix[sim , epoch]) = model_3.evaluate(x_shuffle , t_shuffle)

mean_5, lower_5, upper_5 = [],[],[]
mean_10, lower_10, upper_10 = [],[],[]

ci = 0.95

for i in range (history_3_itself_matrix.shape[1]):

    a = history_3_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_5.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_5.append(Lower)
    upper_5.append(Upper)
    
    a = accuracy_3_itself_matrix[ : , i]

    MEAN = np.mean(a)
    mean_10.append(MEAN)
    std = np.std(a)
    
    Upper = MEAN+ci*std
    Lower = MEAN-ci*std
    lower_10.append(Lower)
    upper_10.append(Upper)

"""
plt.plot(mean_5 , "-b" , label = "mean" , color = "green")
plt.fill_between(list(range(len(mean_5))), upper_5, lower_5, color="green", alpha=0.25)
plt.show()
"""
#%%

fig , ax = plt.subplots(2 , 3)

#Plot the loss functions
ax[0 , 0].plot(mean_1 , "-b" , label = "mean" , color = "green")
ax[0 , 0].fill_between(list(range(len(mean_1))), upper_1, lower_1, color="green", alpha=0.25)
ax[0 , 0].set_title("Model 1")
ax[0 , 0].set_ylabel("Categorical Cross Entropy")
ax[0 , 0].set_ylim(-0.2 , 2.3)

ax[0 , 1].plot(mean_2 , "-b" , label = "mean" , color = "red")
ax[0 , 1].fill_between(list(range(len(mean_2))), upper_2, lower_2, color="red", alpha=0.25)
ax[0 , 1].plot(mean_3 , "-b" , label = "mean" , color = "green")
ax[0 , 1].fill_between(list(range(len(mean_3))), upper_3, lower_3, color="green", alpha=0.25)
ax[0 , 1].set_title("Model 2")
ax[0 , 1].set_ylabel("Categorical Cross Entropy")
ax[0 , 1].set_ylim(-0.2 , 2.3)

ax[0 , 2].plot(mean_4 , "-b" , label = "mean" , color = "green")
ax[0 , 2].fill_between(list(range(len(mean_4))), upper_4, lower_4, color="green", alpha=0.25)
ax[0 , 2].plot(mean_5 , "-b" , label = "mean" , color = "red")
ax[0 , 2].fill_between(list(range(len(mean_5))), upper_5, lower_5, color="red", alpha=0.25)
ax[0 , 2].set_title("Model 3")
ax[0 , 2].set_ylabel("Categorical Cross Entropy")
ax[0 , 2].set_ylim(-0.2 , 2.3)

#Plot the accuracies
ax[1 , 0].plot(mean_6 , "-b" , label = "mean" , color = "green")
ax[1 , 0].fill_between(list(range(len(mean_6))), upper_6, lower_6, color="green", alpha=0.25)
ax[1 , 0].set_title("Model 1")
ax[1 , 0].set_ylabel("Accuracy")
ax[1 , 0].set_ylim(0 , 1.1)

ax[1 , 1].plot(mean_7 , "-b" , label = "mean" , color = "red")
ax[1 , 1].fill_between(list(range(len(mean_7))), upper_7, lower_7, color="red", alpha=0.25)
ax[1 , 1].plot(mean_8 , "-b" , label = "mean" , color = "green")
ax[1 , 1].fill_between(list(range(len(mean_8))), upper_8, lower_8, color="green", alpha=0.25)
ax[1 , 1].set_title("Model 2")
ax[1 , 1].set_ylabel("Accuracy")
ax[1 , 1].set_ylim(0 , 1.1)

ax[1 , 2].plot(mean_9 , "-b" , label = "mean" , color = "green")
ax[1 , 2].fill_between(list(range(len(mean_9))), upper_9, lower_9, color="green", alpha=0.25)
ax[1 , 2].plot(mean_10 , "-b" , label = "mean" , color = "red")
ax[1 , 2].fill_between(list(range(len(mean_10))), upper_10, lower_10, color="red", alpha=0.25)
ax[1 , 2].set_title("Model 3")
ax[1 , 2].set_ylabel("Accuracy")
ax[1 , 2].set_ylim(0 , 1.1)

plt.suptitle("Model losses and accuracies (red without training earlier models, green with)")
fig.show()
