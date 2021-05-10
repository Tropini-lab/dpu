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
from bokeh.layouts import gridplot
from bokeh.io import export_svg
from selenium import webdriver
import copy


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


        #For interest, let's optimize to find what 0 is on that 3d calibration curve..
        b135 = (49000, 57000)
        b90 = (25000,30000)
        bnds = (b135, b90)
        res = minimize(self.opt_minimize,x0=np.array([53000,25250]),args = ([c0, c1, c2, c3, c4, c5]), bounds = bnds)


        #Getting the median of raw od readings over a window...

        df_90 = self.get_raw_df(self.od_90_files[vial_num])
        od_90_baseline_median = np.median(df_90["OD"].tolist()[:window_size])

        df_135 = self.get_raw_df(self.od_135_files[vial_num])
        od_135_baseline_median = np.median(df_135["OD"].tolist()[:window_size])

        factor = [float(res.x[0]-od_135_baseline_median),float(res.x[1]-od_90_baseline_median)]

        return [factor,df_135,df_90]

    def plot_raw_zeroing(self):

        colour_array = ['black', 'rosybrown', 'maroon', 'salmon', 'peru', 'yellow', 'olive', 'lawngreen', 'forestgreen',
                        'aquamarine', 'cyan',
                        'deepskyblue', 'grey', 'blue', 'violet', 'magenta']
        #Initializing bokeh plot:
        b1 = figure(title="OD zeroed with raw", x_axis_label='Time (hours)', y_axis_label='OD')
        b2 = figure(title="OD zeroed", x_axis_label='Time (hours)', y_axis_label='OD')
        b3 = figure(title = "OD 135 degrees Raw", x_axis_label = 'Time (hours)', y_axis_label = 'OD')
        b4 = figure(title = "OD 90 degrees Raw", x_axis_label = 'Time (hours)', y_axis_label = 'OD')

        # os_dict ={0:'230',1:'250',2:'300',3:'350 Turbidostat',5:'400',6:'450',8:'600',9:'700',11:'800',12:'900',13:'1200',15:'350'} # March 18th Trial

        # os_dict = {0:'226',1:'400',2:'300', 3:'500', 5:'800' ,7:'226', 9:'226', 12:'500', 13:'300',15:'800' }  # March 19th Trial

        os_dict = {0:'220 +',1:'220 -',2:'220 +',3:'220 +',4:'',5:'455+',6:'455+',7:'Sterile',8:'455+',9:'455-',10:'',11:'925 +',12:'925 +',13:'925 +',14:'',15:'925 -'}


        #os dict MAY HAVE BEEN WRONG MARCH 18 TRIAL: 11:900,12:1200, 113:350, 15:800 - SEEMS UNLIEKELY THOUGH?
        export_df_rw_zero = pd.DataFrame()      #Create a dataframe to save the OD data in an excel file:
        export_df_od_zero = pd.DataFrame()
        export_df_od_135 = pd.DataFrame()
        export_df_od_90 = pd.DataFrame()
        export_df_time = pd.DataFrame()

        plt.figure()

        # figure
        # for i in [11,12,13,15]:
        for i in range(0,16):

            factor,rw_df_135,rw_df_90 = self.raw_zeroing(60,vial_num = i)
            od_zero_with_od = self.od_zeroing(i,rw_df_135,rw_df_90,window_size=1)

            #Adding on the raw adjustment factor:
            # df_135 = pd.DataFrame()
            # df_90 = pd.DataFrame()
            # df_135["OD"] = copy.deepcopy(rw_df_135["OD"]) + float(factor[0])
            # df_90["OD"] = copy.deepcopy(rw_df_90["OD"]) + float(factor[1])

            #Getting the 3D calibration parameters:
            c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{i}')

            #Using the calibration function to get OD values:
            # od_zero_with_raw = np.real([self.three_dim([float(x), float(y)], c0, c1, c2, c3, c4, c5) for x, y in zip(df_135["OD"].tolist(), df_90["OD"].tolist())])
            time = rw_df_90["Time"].tolist()

            #Ensuring time and od are the same length:
            if len(time)>len(od_zero_with_od):
                time = time[:len(od_zero_with_od)]
            elif len(time)<len(od_zero_with_od):
                od_zero_with_od = od_zero_with_od[:len(time)]

            #Creating an export dataframe:
            # export_df_rw_zero[f'OD  {os_dict.get(i)} mOsm vial {i} '] = od_zero_with_raw
            # export_df_od_zero[f'OD  {os_dict.get(i)} mOsm vial {i}'] = od_zero_with_od
            # export_df_od_135[f'OD 135 {os_dict.get(i)} mOsm vial {i}'] = rw_df_135["OD"].tolist()
            # export_df_od_90[f'OD 90 {os_dict.get(i)} mOsm vial {i}'] = rw_df_90["OD"].tolist()



        #plot against time
            # b1.line(time, medfilt(np.array(od_zero_with_raw),kernel_size=1),line_width=1, color=colour_array[i], legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')
            b2.line(time, medfilt(np.array(od_zero_with_od),kernel_size=7),line_width = 1, color = colour_array[i],legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')
            b3.line(rw_df_135["Time"],rw_df_135["OD"],color = colour_array[i], line_width = 1, legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}' )
            b4.line(rw_df_90["Time"], rw_df_90["OD"], color=colour_array[i], line_width = 1, legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')

            plt.plot(time, medfilt(np.array(od_zero_with_od),kernel_size=11), color = colour_array[i],label =f'Vial {i} + mOsm = {os_dict.get(i)} ')

        # export_df_time['Time (hours)'] = time #Adding time to the dataframe

        # with pd.ExcelWriter(r'C:\Users\erlyall\PycharmProjects\dpu\Mar18_M9_Osmolality_Testing_Data.xlsx') as writer:
        #     export_df_rw_zero.to_excel(writer, sheet_name='OD Raw Zeroed')
        #     export_df_od_zero.to_excel(writer, sheet_name='OD_Zeroed')
        #     export_df_od_135.to_excel(writer,sheet_name='OD 135 Raw')
        #     export_df_od_90.to_excel(writer,sheet_name='OD 90 Raw')
        #     export_df_time.to_excel(writer, sheet_name= 'Time')




        #Adding a interactive legend:
        for b in [b2,b3,b4]:
            b.legend.location = "top_left"
            b.legend.click_policy = "hide"

        grid = gridplot([b2, b3,b4], ncols=1, sizing_mode = 'stretch_width')
        output_file("od_plots.html")
        show(grid)

        #Saving the bokeh plot as a scalable graphics file:


        # export_svg(grid, filename="March18_Osmo_Trials.svg")

        plt.legend()
        plt.title("Feb 5 Batch Trial")
        plt.xlabel('Time (hours)')
        plt.ylabel('OD')
        plt.grid(True)
        plt.show()

    def od_zeroing(self,vial_num, df_135, df_90,window_size = 1):
        i = vial_num

        #Getting the actual OD readings:
        c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{i}')
        od = np.real([self.three_dim([float(x), float(y)], c0, c1, c2, c3, c4, c5) for x, y in
                      zip(df_135["OD"].tolist(), df_90["OD"].tolist())])

        #Getting the average OD readings over the window range:
        baseline_median = np.median(od[:window_size])
        adj_factor = 0- baseline_median
        adj_od = od + adj_factor
        return adj_od


if __name__ == '__main__':
    # browser = webdriver.Chrome()
    cal_dict_90 = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\EricOD90Cal.npy', allow_pickle=True).item()
    cal_dict_135 = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\EricOD135Cal.npy', allow_pickle=True).item()
    cal_3d_params = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Feb43DCal.npy', allow_pickle='TRUE').item()


    od_90_folder = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\April_28_Phage_Osmo_expt\od_90_raw'
    od_135_folder =r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\April_28_Phage_Osmo_expt\od_135_raw'

    RawZero = Zeroing(cal_dict_90,cal_dict_135,cal_3d_params,od_90_folder=od_90_folder,od_135_folder=od_135_folder)
    RawZero.plot_raw_zeroing()
