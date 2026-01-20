###############################
#### Castle experiment PhD ####
###############################

## This experiment attempts to investigate curriculum learning. ##

##Import packages
from psychopy import visual, core, event, gui
import time, numpy, os, pandas
from Abstract_rules import trials as tr
from Abstract_rules import participant_size , participant_orientation , indices

#############################
#### Datafile management ####
#############################

# gui toggle (set to 1 if you want to use the gui)
guitoggle = 1 

##function for messages
def message(message_text = "", response_key = "space", duration = 0, height = None, pos = (0.0, 0.0), color = "black"):
    message_on_screen = visual.TextStim(win, text = "OK")
    message_on_screen.text    = message_text
    message_on_screen.height  = height
    message_on_screen.pos     = pos
    message_on_screen.color   = color
    
    message_on_screen.draw()
    win.flip()
    if duration == 0: # for the welcome and goodbye
        event.waitKeys(keyList = response_key)
    else:
        time.sleep(duration) # for the feedback

# make data folder for storing csv file (won't be saved there automatically)
my_directory    = os.getcwd()
data_directory  = my_directory + "/Data"
new_directory   = my_directory + "/Data/Files"

if not os.path.isdir(new_directory):
    os.mkdir(new_directory)

info = {"First Name" : "Ward" , "Participant Number" : 4 , "Gender" : ["male" , "female" , "other"]}

##Here I check whether or not the file I'm about to make exists. I do this with my participant number because that's the only thing in my file name
already_exists = True
while already_exists:
    ##I present my dialogue box
    myDlg = gui.DlgFromDict(dictionary = info , title = "Castles experiment" , order = ("First Name" , "Participant Number" , "Gender"))
    ##And create my file
    participant = str(info["Participant Number"])
    file_name = new_directory + "/pp" + str(info["Participant Number"])
    ##Then I check whether or not this file exists yet
    if not os.path.isfile(file_name + ".csv"):
        ##If it doesn't, then good (!) and we can exit the loop and start with the rest
        already_exists = False
    else:
        ##If not, then the participant needs to indicate another participant number
        myDlg2 = gui.Dlg(title = "Error")
        myDlg2.addText("This number was already used. Please ask the experimenter to help you to enter a unique number.")
        myDlg2.show()

# instructions function
def display_instructions(file = "instructions.png"):
    instructions.image = file
    instructions.draw()
    win.flip()
    event.clearEvents(eventType = "keyboard")
    event.waitKeys(keyList = "space")

########################
#### Initialization ####
########################

##Define window
win = visual.Window(fullscr = True)
my_clock = core.Clock()

##Define variables for the hints
orientations = numpy.arange(start = 0 , stop = 180 , step = 4.5) #40 orientations
colors = ["blue" , "red"] # 2 colors => do a continuous mapping for this! So from [0 , 0 , 1] to [1 , 0 , 0]
#If I, for color, vary the RGB value for red and blue from 0 to 1 with steps of 0.1, then it's 5 options for each, so 25 options in total... 
red_range = numpy.arange(start = 0.1 , stop = 1 , step = 0.2) # 5 options
blue_range = numpy.arange(start = 0 , stop = 1 , step = 0.2) # 5 options

total_number_of_colors = numpy.arange(start = 0 , stop = 25 , step = 1) # 25 possible colors; crossing of red and blue

size = numpy.arange(start = 0.1, stop = 0.5 , step = 0.04) # 10 sizes

##Read in the images of the castles
castle_directory = my_directory + "/Castle_images"
castle_images = ["/Castle_1.png" , "/Castle_2.png" , "/Castle_3.png" , "/Castle_4.png"]
instructions_directory = my_directory + "/Instructions"

##Define positions of the castles (first one put a little lower for visual reasons)
##Position 2 is for the squares around the castles. Easier for participants to see if they clicked correctly (squares change color whan participants hover over it)
positions = [(-0.6 , -0.05) , (0 , 0) , (0.6 , 0)] #, (0.75 , 0)]
positions_2 = [(-0.6 , 0) , (0 , 0) , (0.6 , 0)] #, (0.75 , 0)]

##Some basic features of the design
n_choices = 40 #This is the amount of time the participant gets to chose a castle
n_trials = 280 #280 #This is the amount of walls the participant encounters; 250 for the practice and 30 for the test

##Some visuals to use later on
Recties         = visual.Rect(win , size = (0.4 , 0.6) , lineWidth = 20, lineColor = "black" , fillColor = None)
Castle          = visual.ImageStim(win, size = (0.3 , 0.5))
Instructions    = visual.TextStim(win , "Select the castle you want to enter" , pos = (0 , 0.7) , color = "black")
chosen_castle   = visual.ImageStim(win , image = castle_directory + castle_images[0] , size = (0.3 , 0.5) , pos = (0 , 0))
Wall            = visual.ImageStim(win , image = castle_directory + "/Wall.jpg" , size = (2 , 2) , pos = (0 , 0))
Door            = visual.ImageStim(win , image = castle_directory + "/Door.png" , pos = (-0.75 , -0.45)  , size = (0.15 , 0.35))
Treasure        = visual.ImageStim(win , image = castle_directory + "/Treasure.jpg" , size = (2 , 2) , pos = (0 , 0))
Placeholder     = visual.Rect(win , fillColor = None , lineColor = "black" , lineWidth = 10 , height = 0.35 , width = 0.15)
image           = visual.ImageStim(win , image = None)
line            = visual.Polygon(win , edges = 2 , lineWidth = 5 , size = 0.2)
hint            = visual.Rect(win , lineWidth = 5)
instructions    = visual.ImageStim(win , size = (2 , 2))
Trial_tracker   = visual.TextStim(win , pos = (0.8 , 0.8))

###########################
## Positions of pictures ##
###########################

##Define the possible positions of the door for castle 1
positions_sample_1 = [(-0.5 , -0.45) , (0.5 , -0.45)]

##Define the possible positions of the door for castle 2
positions_sample_2 = [(-0.75 , -0.45) , (-0.25 , -0.45) , (0.25 , -0.45) , (0.75 , -0.45)]

##Define the possible positions of the door for castle 3
positions_sample_3 = [(-0.85 , -0.45) , (-0.60 , -0.45) , (-0.35 , -0.45) , (-0.1 , -0.45) , (0.1 , -0.45) , (0.35 , -0.45) , (0.6 , -0.45) , (0.85 , -0.45)]

##Define the mouse
my_mouse = event.Mouse()
event.clearEvents(eventType = "mouse")

mouseResponses = [0,0,0]
position = -9999

################################################
## Initialize the file to write the data away ##
################################################

trialnr  = numpy.array(range(1,n_trials+1)) 
empty    = numpy.empty((n_trials,6))
empty.fill(-9999)
participantNr   = numpy.repeat(participant, n_trials)

########################
## Randomize features ##
########################

##Define the participant specific order for the relevant (added) features during each castle 
dimensions = [0 , 1 , 2]
numpy.random.shuffle(dimensions)
criteria = [participant_orientation , 13 , participant_size]

possibilities = ["orientation" , "color" , "size"]

first_castle    = possibilities[int(dimensions[0])]
second_castle   = possibilities[int(dimensions[1])]
third_castle    = possibilities[int(dimensions[2])]

first_feature   = numpy.repeat(first_castle, n_trials)
second_feature  = numpy.repeat(second_castle, n_trials)
third_feature  = numpy.repeat(third_castle, n_trials)

file_participant_size           = numpy.repeat(participant_size, n_trials)
file_participant_orientation    = numpy.repeat(participant_orientation, n_trials)

trials = numpy.column_stack([trialnr , empty , first_feature , second_feature , third_feature , file_participant_size , file_participant_orientation , participantNr])

##Now the file has 13 columns: 
#0: Trial_number: trial_nr; gets added in the loop
#1: Chosen castle (can be the same number if the participant does not exit the castle)
#2: Chosen location 
#3: Correct location
#4: Accuracy
#5: Number of doors; important for test phase!! 
#6: RT
#7: Feature castle 1
#8: Feature castle 2
#9: Feature castle 3
#10: Participant size
#11: Participant orientation
#12: The participant number

##Get the relevant features per castle in the file to see what is doable and what not. 

##Define the file name for each participant and direct it towards the correct directory
file_name = new_directory + "/pp" + str(participant) + ".csv"

##################
## Instructions ##
##################

display_instructions(file = instructions_directory + "/Slide1.png")
display_instructions(file = instructions_directory + "/Slide2.png")
display_instructions(file = instructions_directory + "/Slide3.png")
display_instructions(file = instructions_directory + "/Slide4.png")
display_instructions(file = instructions_directory + "/Slide5.png")
display_instructions(file = instructions_directory + "/Slide6.png")
display_instructions(file = instructions_directory + "/Slide7.png")
display_instructions(file = instructions_directory + "/Slide8.png")
display_instructions(file = instructions_directory + "/Slide9.png")
display_instructions(file = instructions_directory + "/Slide10.png")
display_instructions(file = instructions_directory + "/Slide11.png")
display_instructions(file = instructions_directory + "/Slide12.png")
display_instructions(file = instructions_directory + "/Slide13.png")

###########################
## Start experiment loop ##
###########################

##Initialize trial count
trial_nr = 0

#The -30 is because there are 30 trials for the testing phase, these are included in the n_trials parameter
while trial_nr < (n_trials - 30):
    
    position = -9999
    ##Wait for participant to pick a castle + they cannot click next to a castle
    while numpy.sum(mouseResponses) == 0 or position == -9999:
        position_x = my_mouse.getPos()[0]
        position_y = my_mouse.getPos()[1]
        
        position = -9999
        
        #### Here I draw the castles and put them all on the screen next to each other for the participant to choose #### 
        for index in range(len(castle_images) - 1): 
            #Make squares around the castle images and turn it green when they hover over one
            Recties.pos = positions_2[index]
            Recties.lineColor = None
            Castle.image = castle_directory + castle_images[index]
            Castle.pos = positions[index]
            if float(positions[index][0] - 0.3) <= float(position_x) <= float(positions[index][0] + 0.3) and float(positions[index][1] - 0.3) <= float(position_y) <= float(positions[index][1] + 0.3): 
                Recties.lineColor = "green"
    #            Recties = visual.Rect(win , size = (0.4 , 0.6) , lineWidth = 20 , pos = positions_2[index] , color = "green" , fillColor = None)
                position = index
            Castle.draw()
            Recties.draw()
        
        #Instructions now just say that they need to pick a castle
        Instructions.draw()
        
        Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
        Trial_tracker.draw()
        
        win.flip()
        
        mouseResponses = my_mouse.getPressed()
    
    win.flip()

    ##Then present in the center which one they chose
    if position == 0: 
        chosen_castle.image = castle_directory + castle_images[0]
        Castle_choice = 1
    elif position == 1: 
        chosen_castle.image = castle_directory + castle_images[1]
        Castle_choice = 2
    elif position == 2: 
        chosen_castle.image = castle_directory + castle_images[2]
        Castle_choice = 3
    else: 
        chosen_castle.image = castle_directory + castle_images[3]
        Castle_choice = 4
    
    chosen_castle.draw()
    win.flip()
    time.sleep(1)

    my_mouse = event.Mouse()
    event.clearEvents(eventType = "mouse")

    mouseResponses = [0,0,0]
    position = -9999
    
    ##Initialize before going into the castle
    incorrect = False
    trial = 0
    
    if Castle_choice == 1:
        random_indices = numpy.random.randint(0 , tr.shape[0] , 20)
        ##Want to loop through this until participant gives the wrong response or untill they reach the treasure room
        while not incorrect and trial < 20:
            ##Add the castle choice to the file
            trials[trial_nr , 1] = Castle_choice
            trials[trial_nr , 5] = 2
            
            trial_nr += 1
            
            mouseResponses = [0,0,0]
            my_clock.reset()
            while numpy.sum(mouseResponses) == 0 or position == -9999:
                position_x = my_mouse.getPos()[0]
                position_y = my_mouse.getPos()[1]
                
                position = -9999
                
                Wall.draw()
                
                ##Get the hint features and present the correct hint
                #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4), same for size and orientation
                hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                hint.size = size[int(tr[random_indices[trial] , 2])]
                line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                #Orientation -90 to make the orientation a little more intuitive; otherwise from vertical to vertical
                line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
                hint.draw()
                line.draw()
                
                #Possible door locations are indicated with rectangles. They again turn green when participants hover over it so they know they selected the one they wanted
                #They cannot click next to a door, they have to pick one
                for locations in range(len(positions_sample_1)): 
                    Placeholder.lineColor = "black"
                    if float(positions_sample_1[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_1[locations][0] + 0.1) and float(positions_sample_1[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_1[locations][1] + 0.3): 
                        Placeholder.lineColor = "green"
                        Choice = locations
                        position = locations
                    Placeholder.pos = positions_sample_1[locations]
                    Placeholder.draw()
                
                Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
                Trial_tracker.draw()
                
                win.flip()
                
                mouseResponses = my_mouse.getPressed()
            
            RT = my_clock.getTime()
            Wall.draw()
            
            ##Define the correct answer for that specific trial
            #Dimension[0] is the dimension relevant for the first castle. If it's zero, we need orientation as relevant feature, 1, then it's color and 2 is size
            #So define the correct dimension and then extract that feature
            if dimensions[0] == 0: 
                feature = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[0] == 1: 
                feature = int(tr[random_indices[trial] , 1])
            else: 
                feature = size[int(tr[random_indices[trial] , 2])]
            
            ##In criteria, the relevant criteria per feature are stored. If the feature is smaller than the criterium, the left door is correct, otherwise right
            #Show the door and also define the correct response
            if feature < criteria[dimensions[0]]: 
                Door.pos = positions_sample_1[0]
                Correct = 0
            else: 
                Door.pos = positions_sample_1[1]
                Correct = 1
            
            ##If the participant gets the wrong location, we put incorrect to True, making them exit the castle and chose a new castle on the next trial
            if Choice != Correct: 
                incorrect = True
                Accuracy = 0
                Placeholder.lineColor = "red"
            else: 
                Accuracy = 1
                Placeholder.lineColor = "green"
            
            trials[trial_nr - 1 , 2] = Choice 
            trials[trial_nr - 1 , 3] = Correct
            trials[trial_nr - 1 , 4] = Accuracy
            trials[trial_nr - 1 , 6] = RT
            
            Placeholder.pos = positions_sample_1[Choice]
            Placeholder.draw()
            Door.draw()
            
            Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
            Trial_tracker.draw()
            
            win.flip()
            
            #Show where the correct location was for 2 seconds
            time.sleep(2)
            
            trial += 1
            
            if (trial == 20) and (Accuracy == 1): 
                Treasure.draw()
                win.flip()
                time.sleep(2)
            
            if trial_nr >= (n_trials - 30): 
                break
        
    elif Castle_choice == 2:
        random_indices = numpy.random.randint(0 , tr.shape[0] , 20)
        while not incorrect and trial < 20:
            
            trials[trial_nr , 1] = Castle_choice
            trials[trial_nr , 5] = 4
            
            trial_nr += 1
            mouseResponses = [0,0,0]
            my_clock.reset()
            while numpy.sum(mouseResponses) == 0 or position == -9999:
                position_x = my_mouse.getPos()[0]
                position_y = my_mouse.getPos()[1]
                
                position = -9999
                
                Wall.draw()
                
                #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4)
                hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                hint.size = size[int(tr[random_indices[trial] , 2])]
                line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
                hint.draw()
                line.draw()
                
                for locations in range(len(positions_sample_2)):
                    Placeholder.lineColor = "black"
                    if float(positions_sample_2[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_2[locations][0] + 0.1) and float(positions_sample_2[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_2[locations][1] + 0.3): 
                        Placeholder.lineColor = "green"
                        Choice = locations
                        position = locations
                    Placeholder.pos = positions_sample_2[locations]
                    Placeholder.draw()
                
                Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
                Trial_tracker.draw()
                
                win.flip()
                
                mouseResponses = my_mouse.getPressed()
            
            RT = my_clock.getTime()
            Wall.draw()
            
            
            if dimensions[0] == 0: 
                feature = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[0] == 1: 
                feature = int(tr[random_indices[trial] , 1])
            else: 
                feature = size[int(tr[random_indices[trial] , 2])]
            
            if dimensions[1] == 0: 
                feature_2 = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[1] == 1: 
                feature_2 = int(tr[random_indices[trial] , 1])
            else: 
                feature_2 = size[int(tr[random_indices[trial] , 2])]
            
            if feature < criteria[dimensions[0]]: 
                if feature_2 < criteria[dimensions[1]]: 
                    Door.pos = positions_sample_2[0]
                    Correct = 0
                else: 
                    Door.pos = positions_sample_2[1]
                    Correct = 1
            else: 
                if feature_2 < criteria[dimensions[1]]: 
                    Door.pos = positions_sample_2[2]
                    Correct = 2
                else: 
                    Door.pos = positions_sample_2[3]
                    Correct = 3
            
            if Choice != Correct: 
                incorrect = True
                Accuracy = 0
                Placeholder.lineColor = "red"
            else: 
                Accuracy = 1
                Placeholder.lineColor = "green"
            
            trials[trial_nr - 1 , 2] = Choice 
            trials[trial_nr - 1 , 3] = Correct
            trials[trial_nr - 1 , 4] = Accuracy
            trials[trial_nr - 1 , 6] = RT
            
            Placeholder.pos = positions_sample_2[Choice]
            Placeholder.draw()
            Door.draw()
            
            Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
            Trial_tracker.draw()
            
            win.flip()
            time.sleep(2)
            
            trial += 1
            
            
            if (trial == 20) and (Accuracy == 1): 
                Treasure.draw()
                win.flip()
                time.sleep(2)
            
            if trial_nr >= (n_trials - 30): 
                break
        
    elif Castle_choice == 3:
        random_indices = numpy.random.randint(0 , tr.shape[0] , 20)
        while not incorrect and trial < 20:
            
            trials[trial_nr , 1] = Castle_choice
            trials[trial_nr , 5] = 8
            
            mouseResponses = [0,0,0]
            my_clock.reset()
            trial_nr += 1
            while numpy.sum(mouseResponses) == 0 or position == -9999:
                position_x = my_mouse.getPos()[0]
                position_y = my_mouse.getPos()[1]
                
                position = -9999
                
                Wall.draw()
                
                #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4)
                hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                hint.size = size[int(tr[random_indices[trial] , 2])]
                line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
                line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
                hint.draw()
                line.draw()
                
                for locations in range(len(positions_sample_3)):
                    Placeholder.lineColor = "black"
                    if float(positions_sample_3[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_3[locations][0] + 0.1) and float(positions_sample_3[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_3[locations][1] + 0.3): 
                        Placeholder.lineColor = "green"
                        Choice = locations
                        position = locations
                    Placeholder.pos = positions_sample_3[locations]
                    Placeholder.draw()
                
                Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
                Trial_tracker.draw()
                
                win.flip()
                
                mouseResponses = my_mouse.getPressed()
            
            RT = my_clock.getTime()
            Wall.draw()
            
            if dimensions[0] == 0: 
                feature = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[0] == 1: 
                feature = int(tr[random_indices[trial] , 1])
            else: 
                feature = size[int(tr[random_indices[trial] , 2])]
            
            if dimensions[1] == 0: 
                feature_2 = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[1] == 1: 
                feature_2 = int(tr[random_indices[trial] , 1])
            else: 
                feature_2 = size[int(tr[random_indices[trial] , 2])]
            
            if dimensions[2] == 0: 
                feature_3 = orientations[int(tr[random_indices[trial] , 0])]
            elif dimensions[2] == 1: 
                feature_3 = int(tr[random_indices[trial] , 1])
            else: 
                feature_3 = size[int(tr[random_indices[trial] , 2])]
            
            if feature < criteria[dimensions[0]]: 
                if feature_2 < criteria[dimensions[1]]: 
                    if feature_3 < criteria[dimensions[2]]:
                        Door.pos = positions_sample_3[0]
                        Correct = 0
                    else: 
                        Door.pos = positions_sample_3[1]
                        Correct = 1
                else: 
                    if feature_3 < criteria[dimensions[2]]:
                        Door.pos = positions_sample_3[2]
                        Correct = 2
                    else: 
                        Door.pos = positions_sample_3[3]
                        Correct = 3
            else: 
                if feature_2 < criteria[dimensions[1]]: 
                    if feature_3 < criteria[dimensions[2]]:
                        Door.pos = positions_sample_3[4]
                        Correct = 4
                    else: 
                        Door.pos = positions_sample_3[5]
                        Correct = 5
                else: 
                    if feature_3 < criteria[dimensions[2]]:
                        Door.pos = positions_sample_3[6]
                        Correct = 6
                    else: 
                        Door.pos = positions_sample_3[7]
                        Correct = 7
            
            if Choice != Correct: 
                incorrect = True
                Accuracy = 0
                Placeholder.lineColor = "red"
            else: 
                Accuracy = 1
                Placeholder.lineColor = "green"
            
            trials[trial_nr - 1 , 2] = Choice 
            trials[trial_nr - 1 , 3] = Correct
            trials[trial_nr - 1 , 4] = Accuracy
            trials[trial_nr - 1 , 6] = RT
            
            Placeholder.pos = positions_sample_3[Choice]
            Placeholder.draw()
            Door.draw()
            
            Trial_tracker.text = "Trial " + str(trial_nr) + "/" + str(n_trials - 30)
            Trial_tracker.draw()
            
            win.flip()
            time.sleep(2)
            
            trial += 1
            
            if (trial == 20) and (Accuracy == 1): 
                Treasure.draw()
                win.flip()
                time.sleep(2)
            
            if trial_nr >= (n_trials - 30): 
                break
        
    else: 
        ##Add the castle choice to the file
        trials[trial_nr , 1] = Castle_choice
        
        trial_nr += 1
        print(trial_nr)
        warning = visual.TextStim(win , text = "This is the test castle, first train!!" , color = "black")
        warning.draw()
        win.flip()
        time.sleep(2)


#################
## Test Castle ##
#################

##After 250 trials, we go to the test castle

#30 walls, 10 from each castle as a test, we randomly shuffle the different castles/difficulties 
#Shuffled order is in indices

test_castle = visual.ImageStim(win , image = castle_directory + castle_images[3] , size = (0.3 , 0.5) , pos = (0 , 0))
test_text = visual.TextStim(win , text = "This is the test castle, good luck!" , pos = (0 , 0.7) , color = "black")

test_castle.draw()
test_text.draw()
win.flip()
time.sleep(2)

#Get random numbers for selection of the hints (rules should be clear, so it can be different ones as in the learning phase)
random_indices = numpy.random.randint(0 , tr.shape[0] , 30)

for trial in range(len(indices)): 
    
    trials[trial_nr , 1] = 4
    trials[trial_nr , 5] = indices[trial] #Indices is defined in the abstract_rules script. 
    #It's a list with the numbers 2, 4 and 8, all 10 times. 2, 4 and 8 stands for the number of doors in the trial 
    #2 being the easy castle, 4 the medium castle and 8 the hard one
    
    if indices[trial] == 2: 
        trial_nr += 1
        
        mouseResponses = [0,0,0]
        my_clock.reset()
        while numpy.sum(mouseResponses) == 0 or position == -9999:
            position_x = my_mouse.getPos()[0]
            position_y = my_mouse.getPos()[1]
            
            position = -9999
            
            Wall.draw()
            
            ##Get the hint features and present the correct hint
            #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4), same for size and orientation
            hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            hint.size = size[int(tr[random_indices[trial] , 2])]
            line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            #Orientation -90 to make the orientation a little more intuitive; otherwise from vertical to vertical
            line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
            hint.draw()
            line.draw()
            
            #Possible door locations are indicated with rectangles. They again turn green when participants hover over it so they know they selected the one they wanted
            #They cannot click next to a door, they have to pick one
            for locations in range(len(positions_sample_1)): 
                Placeholder.lineColor = "black"
                if float(positions_sample_1[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_1[locations][0] + 0.1) and float(positions_sample_1[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_1[locations][1] + 0.3): 
                    Placeholder.lineColor = "green"
                    Choice = locations
                    position = locations
                Placeholder.pos = positions_sample_1[locations]
                Placeholder.draw()
            
            win.flip()
            
            mouseResponses = my_mouse.getPressed()
        
        RT = my_clock.getTime()
        Wall.draw()
        
        ##Define the correct answer for that specific trial
        #Dimension[0] is the dimension relevant for the first castle. If it's zero, we need orientation as relevant feature, 1, then it's color and 2 is size
        #So define the correct dimension and then extract that feature
        if dimensions[0] == 0: 
            feature = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[0] == 1: 
            feature = int(tr[random_indices[trial] , 1])
        else: 
            feature = size[int(tr[random_indices[trial] , 2])]
        
        ##In criteria, the relevant criteria per feature are stored. If the feature is smaller than the criterium, the left door is correct, otherwise right
        #Show the door and also define the correct response
        if feature < criteria[dimensions[0]]: 
            Door.pos = positions_sample_1[0]
            Correct = 0
        else: 
            Door.pos = positions_sample_1[1]
            Correct = 1
        
        ##If the participant gets the wrong location, we put incorrect to True, making them exit the castle and chose a new castle on the next trial
        if Choice != Correct: 
            Accuracy = 0
        else: 
            Accuracy = 1
        
        trials[trial_nr - 1 , 2] = Choice 
        trials[trial_nr - 1 , 3] = Correct
        trials[trial_nr - 1 , 4] = Accuracy
        trials[trial_nr - 1 , 6] = RT
        
        ##Not present the door anymore as I don't want them to learn anymore here, just a test phase 
        #Door.draw()
        win.flip()
        
        #Show where the correct location was for 2 seconds
        time.sleep(2)
        
        trial += 1
        
    elif indices[trial] == 4:
        
        trials[trial_nr , 1] = 4
        
        trial_nr += 1
        mouseResponses = [0,0,0]
        my_clock.reset()
        
        while numpy.sum(mouseResponses) == 0 or position == -9999:
            position_x = my_mouse.getPos()[0]
            position_y = my_mouse.getPos()[1]
            
            position = -9999
            
            Wall.draw()
            
            #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4)
            hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            hint.size = size[int(tr[random_indices[trial] , 2])]
            line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
            hint.draw()
            line.draw()
            
            for locations in range(len(positions_sample_2)):
                Placeholder.lineColor = "black"
                if float(positions_sample_2[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_2[locations][0] + 0.1) and float(positions_sample_2[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_2[locations][1] + 0.3): 
                    Placeholder.lineColor = "green"
                    Choice = locations
                    position = locations
                Placeholder.pos = positions_sample_2[locations]
                Placeholder.draw()
            
            win.flip()
            
            mouseResponses = my_mouse.getPressed()
        
        RT = my_clock.getTime()
        Wall.draw()
        
        if dimensions[0] == 0: 
            feature = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[0] == 1: 
            feature = int(tr[random_indices[trial] , 1])
        else: 
            feature = size[int(tr[random_indices[trial] , 2])]
        
        if dimensions[1] == 0: 
            feature_2 = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[1] == 1: 
            feature_2 = int(tr[random_indices[trial] , 1])
        else: 
            feature_2 = size[int(tr[random_indices[trial] , 2])]
        
        if feature < criteria[dimensions[0]]: 
            if feature_2 < criteria[dimensions[1]]: 
                Door.pos = positions_sample_2[0]
                Correct = 0
            else: 
                Door.pos = positions_sample_2[1]
                Correct = 1
        else: 
            if feature_2 < criteria[dimensions[1]]: 
                Door.pos = positions_sample_2[2]
                Correct = 2
            else: 
                Door.pos = positions_sample_2[3]
                Correct = 3
        
        if Choice != Correct: 
            Accuracy = 0
        else: 
            Accuracy = 1
        
        trials[trial_nr - 1 , 2] = Choice 
        trials[trial_nr - 1 , 3] = Correct
        trials[trial_nr - 1 , 4] = Accuracy
        trials[trial_nr - 1 , 6] = RT
        
        #Door.draw()
        win.flip()
        time.sleep(2)
        
        trial += 1
        
    elif indices[trial] == 8:
        trials[trial_nr , 1] = 4
        
        mouseResponses = [0,0,0]
        trial_nr += 1
        my_clock.reset()
        
        while numpy.sum(mouseResponses) == 0 or position == -9999:
            position_x = my_mouse.getPos()[0]
            position_y = my_mouse.getPos()[1]
            
            position = -9999
            
            Wall.draw()
            
            #Define the color; RGB, with red the 4th column (index 3) and blue the 5th row (index 4)
            hint.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            hint.size = size[int(tr[random_indices[trial] , 2])]
            line.lineColor = [tr[random_indices[trial] , 3] , 0 , tr[random_indices[trial] , 4] ]
            line.ori = orientations[int(tr[random_indices[trial] , 0])] - 90
            hint.draw()
            line.draw()
            
            for locations in range(len(positions_sample_3)):
                Placeholder.lineColor = "black"
                if float(positions_sample_3[locations][0] - 0.1) <= float(position_x) <= float(positions_sample_3[locations][0] + 0.1) and float(positions_sample_3[locations][1] - 0.3) <= float(position_y) <= float(positions_sample_3[locations][1] + 0.3): 
                    Placeholder.lineColor = "green"
                    Choice = locations
                    position = locations
                Placeholder.pos = positions_sample_3[locations]
                Placeholder.draw()
            
            win.flip()
            
            mouseResponses = my_mouse.getPressed()
        
        RT = my_clock.getTime()
        Wall.draw()
        
        if dimensions[0] == 0: 
            feature = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[0] == 1: 
            feature = int(tr[random_indices[trial] , 1])
        else: 
            feature = size[int(tr[random_indices[trial] , 2])]
        
        if dimensions[1] == 0: 
            feature_2 = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[1] == 1: 
            feature_2 = int(tr[random_indices[trial] , 1])
        else: 
            feature_2 = size[int(tr[random_indices[trial] , 2])]
        
        if dimensions[2] == 0: 
            feature_3 = orientations[int(tr[random_indices[trial] , 0])]
        elif dimensions[2] == 1: 
            feature_3 = int(tr[random_indices[trial] , 1])
        else: 
            feature_3 = size[int(tr[random_indices[trial] , 2])]
        
        if feature < criteria[dimensions[0]]: 
            if feature_2 < criteria[dimensions[1]]: 
                if feature_3 < criteria[dimensions[2]]:
                    Door.pos = positions_sample_3[0]
                    Correct = 0
                else: 
                    Door.pos = positions_sample_3[1]
                    Correct = 1
            else: 
                if feature_3 < criteria[dimensions[2]]:
                    Door.pos = positions_sample_3[2]
                    Correct = 2
                else: 
                    Door.pos = positions_sample_3[3]
                    Correct = 3
        else: 
            if feature_2 < criteria[dimensions[1]]: 
                if feature_3 < criteria[dimensions[2]]:
                    Door.pos = positions_sample_3[4]
                    Correct = 4
                else: 
                    Door.pos = positions_sample_3[5]
                    Correct = 5
            else: 
                if feature_3 < criteria[dimensions[2]]:
                    Door.pos = positions_sample_3[6]
                    Correct = 6
                else: 
                    Door.pos = positions_sample_3[7]
                    Correct = 7
        
        if Choice != Correct: 
            Accuracy = 0
        else: 
            Accuracy = 1
        
        trials[trial_nr - 1 , 2] = Choice 
        trials[trial_nr - 1 , 3] = Correct
        trials[trial_nr - 1 , 4] = Accuracy
        trials[trial_nr - 1 , 6] = RT
        
        #Door.draw()
        win.flip()
        time.sleep(2)
        
        trial += 1

##Then save the data in a csv file
trials = pandas.DataFrame.from_records(trials)
trials.columns = ["Trial_number", "Chosen_castle", "Chosen_location", "Correct_location", "Accuracy", "Number_of_doors", "RT" , 
                   "First_feature" , "Second_feature" , "Third_feature" , "Participant_size" , "Participant_orientation" , "Participant_number"]
trials.to_csv(path_or_buf = file_name, index = False)
