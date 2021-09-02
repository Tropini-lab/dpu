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

x =3
EXP_NAME = 'July_9_negative_cntrl_expt'
save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'
file_name = f'vial{x}_pump_log.txt' # "vial{1}_pump_log.txt".format(x)
file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)
data = np.genfromtxt(file_path, delimiter=',')
#Getting total volumes of pumps:

print(len(data))
print("First few lines",data[1][0])

last_pump = data[len(data) - 1][0]

print("Last pump time:", last_pump)
