from psychopy import visual
import numpy as np
import time , pandas


orientations = np.arange(start = 0 , stop = 180 , step = 4.5) #40 orientations
colors = ["blue" , "red"] # 2 colors => do a continuous mapping for this! So from [0 , 0 , 1] to [1 , 0 , 0]
#If I, for color, vary the RGB value for red and blue from 0 to 1 with steps of 0.1, then it's 5 options for each, so 25 options in total... 
red_range = np.arange(start = 0.1 , stop = 1 , step = 0.2) # 5 options
blue_range = np.arange(start = 0 , stop = 1 , step = 0.2) # 5 options

total_number_of_colors = np.arange(start = 0 , stop = 25 , step = 1) # 25 possible colors; crossing of red and blue

size = np.arange(start = 0.1, stop = 0.5 , step = 0.04) # 10 sizes

norientations = len(orientations)
ncolors = len(total_number_of_colors)
nsizes = len(size)

nUnique = norientations * ncolors * nsizes

UniqueTrials = np.array(range(nUnique))

orientations_2 = np.floor(UniqueTrials / (ncolors * nsizes))
colors_2 = np.floor(UniqueTrials / (nsizes)) % (ncolors)
sizes_2 = np.floor(UniqueTrials % (nsizes))

trials = np.column_stack([orientations_2 , colors_2 , sizes_2])
## 10 000 possible combinations for stimuli if we use 5 dimensions
#0 : orientation
#1 : colors
#2 : sizes

###################
## Get variables ##
###################

#So the orientation should be between 30 and 150 AND should not be 90
participant_orientation = 90

while participant_orientation == 90: 
    participant_orientation = np.random.choice(orientations)
    
    if participant_orientation < 30: 
        participant_orientation = 90
    if participant_orientation > 150: 
        participant_orientation = 90

##Say 25 possible options for colors; say 0 - 12 we label as blue and 13-25 we label as red
zero_list = np.repeat(0 , nUnique)

trials = np.column_stack([trials , zero_list , zero_list])
#0 : orientation
#1 : colors
#2 : sizes
#3 : red 
#4 : blue

lower_bound = np.arange(0.2 , 0.45 , 0.05)
upper_bound = np.arange(0.55 , 0.8 , 0.05)

for i in range(trials.shape[0]): 
    if trials[i , 1] <= 12: #More blue than red here
        trials[i , 3] = np.random.choice(lower_bound)
        trials[i , 4] = np.random.choice(upper_bound)
    else: #More red than blue here
        trials[i , 3] = np.random.choice(upper_bound)
        trials[i , 4] = np.random.choice(lower_bound)

##Select a boundary for each participant
##I would not do a random bound per participant here tbh, just go more red vs more blue... 

##Select random boundary for size
#10 sizes; select one between 3 and 7; can be the middle, idc like for orientation; less clear! 
#0.1 - 0.5 with increments of 0.04; 0.22 - 0.38

participant_size = 0
while (participant_size < 0.22) or (participant_size > 0.38): 
    participant_size = np.random.choice(size)

indices = [2 , 4 , 8]
indices = np.tile(indices , 10)
np.random.shuffle(indices)
