import numpy as np
import json


#If you used the standared evovler OD calibration reccomended on the website
#And you select that calibration using the eVOVLER GUI
#The od_cal.json file on the computer connected to the evolver will be that calibration

#We are going to save a slightly modified version of this file a) for reference, and b) to use
#in the eric_evolver_exp_graphing.py script.
#Without saving this calibration, you won't be able to use the eric_evolver_exp_graphing.py script, or plot old data.

#Loading the od_cal.json file:
#TODO: Make sure you change the username to the correct filpath!!
new_cal = open(r'C:\Users\eric1\PycharmProjects\dpu\experiment\template\od_cal.json')
new_cal_data = json.load(new_cal)
print("Json file")
print(new_cal_data.get('coefficients'))

#Making a dictionary to fit eric_ecolver_exp_graphing.py format
new_calibration = {}
for i in range(16):
    new_calibration[f'Vial{i}'] = new_cal_data.get('coefficients')[i]

#Saving this dictionary as a numpy file:
#Change the filename to something informative.. like calibration_Jan_14_2022.npy
#Make sure to change the username (erlyall) to whatever your user name is.
print("New calibration \n", new_calibration)

#TODO: Make sure you change the username to the correct filepath!!
#You'll have to reference this file in the eric_evolver_exp_graphing folder- see the code or notion for details.
np.save(r'C:\Users\eric1\PycharmProjects\dpu\Eric_Graphing\calibration_DATE_3dcal.npy',new_calibration)