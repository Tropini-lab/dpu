import numpy as np
import os

#Getting filepath:
file_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\Feb_19_turb3_expt\pump_log\vial0_pump_log.txt'
data = np.genfromtxt(file_path, delimiter=',')
print(data)
print(len(data))

# save_path = r'C:\Users\erlyall\Desktop\eVOLVER_CODE\Practice text files'
# EXP_NAME = 'FebLearn'
# #Creating a new file..
# for x in range(0,16):
#
#     file_name = "vial{0}_pump_log.txt".format(x)
#     file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)
#
#     text_file = open(file_path, "a+")
#     text_file.write("{0},{1}\n".format(.1,x))
#     text_file.close()