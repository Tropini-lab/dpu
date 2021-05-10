import numpy as np
import pandas as pd
import os
# for x in range(0,16):
#     EXP_NAME = 'April_28_Phage_Osmo_expt'
#     save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'
#     file_name = "vial{0}_pump_log.txt".format(x)
#     file_path = os.path.join(save_path, EXP_NAME,'pump_log', file_name)
#     # print(file_path)
#     data = np.genfromtxt(file_path, delimiter=',')
#     print(f'Vial {x}', " Amount of pumps ", len(data)-2)

x =5
EXP_NAME = 'April_28_Phage_Osmo_expt'
save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'
file_name = "vial{0}_pump_log.txt".format(x)
file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)
data = np.genfromtxt(file_path, delimiter=',')
#Getting total volumes of pumps:
dil_times = data[1]
dil_vols = [.77 * el for el in dil_times]
total_dil_vol = np.sum(np.array(dil_vols))
print("total dilution volume: ", total_dil_vol)