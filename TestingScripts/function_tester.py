import numpy as np
import pandas as pd
import os
x = 1
EXP_NAME = 'April_28_Phage_Osmo_expt'
save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'
file_name = "vial{0}_pump_log.txt".format(x)
file_path = os.path.join(save_path, EXP_NAME,'pump_log', file_name)
print(file_path)
data = np.genfromtxt(file_path, delimiter=',')
print("Amount of pumps ", len(data)-2)

