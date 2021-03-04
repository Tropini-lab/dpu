import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy.signal import medfilt
import os
from scipy.optimize import minimize
import pandas as pd
import pickle
import math
from bokeh.plotting import figure, output_file, show

class Zeroing:

    def __init__(self, cal_dict_90,cal_dict_135,cal_3d_params,od_90_folder,od_135_folder):
        self.cal_dict_90 = cal_dict_90
        self.cal_dict_135 = cal_dict_135
        self.cal_3d_params = cal_3d_params
        self.od_90_files = self.create_od_file_list(od_90_folder)
        self.od_135_files = self.create_od_file_list(od_135_folder)

    def create_od_file_list(self,od_folder):
        #Takes in a od90 or od 135 folder, and returns all the file names in a list in order of vials.
        od_files_list = []
        for path, subdirs, files in os.walk(od_folder):
            for name in files:
                x = os.path.join(path, name)
                od_files_list.append(x)
        od_files_list.sort(key=self.file_num)
        return od_files_list # a list of filenames containing the raw od data, in order of the vials.

    def opt_minimize(self,x,cpar):
        #where x =[od135,od90]
        c0, c1, c2, c3, c4, c5 = cpar
        od = self.three_dim(x,c0, c1, c2, c3, c4, c5)
        return abs(od)

    def file_num(self,filename):

        start_index = int(str.index(filename, "vial")+4)
        end_index = int(str.index(filename, "_od"))
        filenumber = int(filename[start_index: end_index])
        return filenumber

    def get_raw_df(self,filepath):
        df= pd.read_csv(filepath)
        cols = list(df.columns)
        df = df.rename(columns = {cols[0]: "Time", cols[1]: "OD"})
        return df

    def three_dim(self,data, c0, c1, c2, c3, c4, c5):
        x = data[0]  # OD 135 data
        y = data[1]  # OD 90 data
        z = float(c0 + c1 * y + c2 * x + c3 * y ** 2 + c4 * x * y + c5 * x ** 2)
        return z

    def raw_zeroing(self,window_size, vial_num):

        #Two zero vials were used from the calibrations, we take the average raw value of the two of them.
        cal_zero_90  = np.mean((self.cal_dict_90["medians"][vial_num][:2]))
        cal_zero_135 = np.mean((self.cal_dict_135["medians"][vial_num][:2]))

        #For interest, let's see what the 3D calibration thinks these are...
        c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{vial_num}')
        print("Vial", vial_num)
        print("Cal dict 90 \n",self.cal_dict_90)
        print("Cal dict 135\n", cal_dict_135)
        print("Cal 135", cal_zero_135, "Cal 90", cal_zero_90)
        print("OD of what should be 0 on calibration:", self.three_dim([cal_zero_135,cal_zero_90],c0, c1, c2, c3, c4, c5))



        #For interest, let's optimize to find what 0 is on that 3d calibration curve..
        b135 = (49000, 57000)
        b90 = (25000,30000)
        bnds = (b135, b90)
        res = minimize(self.opt_minimize,x0=np.array([53000,25250]),args = ([c0, c1, c2, c3, c4, c5]), bounds = bnds)
        print("Optimized raw values to get od of 0", res.x)
        print("Backcalc optimized values:", self.three_dim([res.x[0],res.x[1]],c0, c1, c2, c3, c4, c5))


        #Getting the median of raw od readings over a window...

        df_90 = self.get_raw_df(self.od_90_files[vial_num])
        od_90_baseline_median = np.median(df_90["OD"].tolist()[:window_size])

        df_135 = self.get_raw_df(self.od_135_files[vial_num])
        od_135_baseline_median = np.median(df_135["OD"].tolist()[:window_size])



        print("Expt 135", od_135_baseline_median, "Expt 90", od_90_baseline_median)
        print("OD of what should be close to zero on experiment:", self.three_dim([od_135_baseline_median,od_90_baseline_median],c0, c1, c2, c3, c4, c5))

        factor = [float(res.x[0]-od_135_baseline_median),float(res.x[1]-od_90_baseline_median)]

        return [factor,df_135,df_90]

    def plot_raw_zeroing(self):

        colour_array = ['black', 'rosybrown', 'maroon', 'salmon', 'peru', 'yellow', 'olive', 'lawngreen', 'forestgreen',
                        'aquamarine', 'cyan',
                        'deepskyblue', 'grey', 'blue', 'violet', 'magenta']
        #Initializing bokeh plot:
        bok_plot = figure(title="OD zeroed with raw", x_axis_label='Time (hours)', y_axis_label='OD')

        for i in range(16):

            factor,df_135,df_90 = self.raw_zeroing(101,vial_num = i)

            #Adding on the raw adjustment factor:
            df_135["OD"] = df_135["OD"] + float(factor[0])
            df_90["OD"] = df_90["OD"] + float(factor[1])

            #Getting the 3D calibration parameters:
            c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{i}')

            #Using the calibration function to get OD values:
            od = np.real([self.three_dim([float(x), float(y)], c0, c1, c2, c3, c4, c5) for x, y in zip(df_135["OD"].tolist(), df_90["OD"].tolist())])
            time = df_90["Time"].tolist()

            #plot against time
            bok_plot.line(time, medfilt(np.array(od),kernel_size=7),line_width=2, color=colour_array[i], legend=f'Vial{i}')

        output_file("od_plots.html")
        show(bok_plot)

if __name__ == '__main__':
    cal_dict_90 = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\EricOD90Cal.npy', allow_pickle=True).item()
    cal_dict_135 = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\EricOD135Cal.npy', allow_pickle=True).item()
    cal_3d_params = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Feb43DCal.npy', allow_pickle='TRUE').item()


    od_90_folder = r'C:\Users\erlyall\Desktop\eVOLVER_CODE\Old Experimental Data\Feb_8_Batch_expt_zeroed\od_90_raw'
    od_135_folder = r'C:\Users\erlyall\Desktop\eVOLVER_CODE\Old Experimental Data\Feb_8_Batch_expt_zeroed\od_135_raw'

    RawZero = Zeroing(cal_dict_90,cal_dict_135,cal_3d_params,od_90_folder=od_90_folder,od_135_folder=od_135_folder)
    RawZero.plot_raw_zeroing()




#
# for i in range(0,16):
#     medians = cal_dict["medians"]
#     measured_data = cal_dict["measured_data"]
#     c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{1}')
#
# high_data = np.array([float(32157),float(26200)]) #135, 90
# initial_data = np.array([float(51792),float(28620)])# 135, 90
# true_zero_data = np.array([np.mean([float(53078),float(51236)]),np.mean([float(27476),float(27784)])]) #135,90
#
#
# #Outputting what high data would be via zeroing OD values:
# initial_od = three_dim(list(initial_data),c0, c1, c2, c3, c4, c5)
# high_od = three_dim(list(high_data),c0, c1, c2, c3, c4, c5)
# zero_by_od = high_od-initial_od
#
# #Outputting what hight data would be via zeroing raw values:
# input = np.add(np.subtract(true_zero_data,initial_data),high_data)
# zero_by_raw = three_dim(input,c0, c1, c2, c3, c4, c5)
#
# print("Calibrating by optical density:", zero_by_od, "Calibrating by raw", zero_by_raw)



