import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import json

#Testing new token in the function tester.
#Testing new token with a pycharm push

#Testing transfer to tropini lab org on bash

#Testing transfer to tropini lab org on pycharm

#Testing tropini lab commit from lab computer pycharm
x = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Feb43DCal.npy',allow_pickle='TRUE').item()

y = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Eric_Apr24_20223dcal.npy',allow_pickle='TRUE').item()

c0, c1, c2, c3, c4, c5 = x.get(f'Vial{0}')

print(c0,c1,c2,c3,c4,c5)

c00, c10, c20, c30, c40, c50 = y.get(f'Vial{0}')
print(c00,c10,c20,c30,c40,c50)

# new_cal = open(r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\od_cal.json')
# new_cal_data = json.load(new_cal)
# print(new_cal_data.get('coefficients')[0])
#
# #Making a dictionary to fit text_writing_practic.py format, and saving it as a numpy file...
# apr_24_2022_cal = {}
# for i in range(16):
#     apr_24_2022_cal[f'Vial{i}'] = new_cal_data.get('coefficients')[i]
# np.save(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Eric_Apr24_20223dcal.npy',apr_24_2022_cal)
