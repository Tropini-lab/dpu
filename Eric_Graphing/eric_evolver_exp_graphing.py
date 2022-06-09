import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
from scipy.optimize import minimize
import pandas as pd
import pickle
import math
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.io import export_svg
# from selenium import webdriver

class Zeroing:

    def __init__(self, cal_3d_params, od_90_folder, od_135_folder):
        self.cal_3d_params = cal_3d_params
        self.od_90_files = self.create_od_file_list(od_90_folder)
        self.od_135_files = self.create_od_file_list(od_135_folder)

    def create_od_file_list(self,od_folder):
        """
        This function takes all the files in a folder, and outputs a list in order of vial number rather than the current
        alphabetical order (0,1,10,11..... )
        :param od_folder: A folder containing optical density readings (either OD 135 raw or OD 90 raw)
        :return: A list of files in that folder, in order of vial number from vial 0 to vial 15.
        """
        #Takes in a od90 or od 135 folder, and returns all the file names in a list in order of vials.
        od_files_list = []
        for path, subdirs, files in os.walk(od_folder):
            for name in files:
                x = os.path.join(path, name)
                od_files_list.append(x)
        od_files_list.sort(key=self.file_num)
        return od_files_list # a list of filenames containing the raw od data, in order of the vials.

    def opt_minimize(self,x,cpar):
        """
        This function is used during raw zeroing to help find the raw OD 90, OD 135 values that correspond to a calculated OD of 0.
        The scipy optimize.minimize funciton tries to minimize the output of this function by changing x.
        :param x: An list containing [od135raw, od90raw] in that order.
        :param cpar: A list of the 5 3d calibration parameters from a given vial in order from c0 to c5
        :return: calculated optical density
        """
        #where x =[od135,od90]
        c0, c1, c2, c3, c4, c5 = cpar
        od = self.three_dim(x,c0, c1, c2, c3, c4, c5)
        return abs(od)

    def file_num(self,filename):
        """
        Looks at the filename, and returns the vial number
        :param filename: A filename for raw OD90 or OD135 data for a given file
        :return: the vial number this came from (0 through 15)
        """
        start_index = int(str.index(filename, "vial")+4)
        end_index = int(str.index(filename, "_od"))
        # new_end_index = int(filename.find("_od", filename.find("_od")+1)) # if looking at calibration from august
        filenumber = int(filename[start_index: end_index])
        return filenumber

    def get_raw_df(self,filepath):
        """
        Creates a pandas dataframe from a text file containing time and optical densites
        :param filepath: The filepath to a raw OD 90 or raw OD 130 file for a given vial
        :return: A pandas dataframe containing read times and raw optical densities.
        """
        df= pd.read_csv(filepath)
        cols = list(df.columns)
        df = df.rename(columns = {cols[0]: "Time", cols[1]: "OD"})
        return df

    def three_dim(self,data, c0, c1, c2, c3, c4, c5):
        """

        :param data: array containing OD 135 data in the first position, and OD 90 data in the second position.
        :param c0: constant from the 3D calibration for a given vial
        :param c1: constant from the 3D calibration for a given vial
        :param c2: constant from the 3D calibration for a given vial
        :param c3: constant from the 3D calibration for a given vial
        :param c4: constant from the 3D calibration for a given vial
        :param c5: constant from the 3D calibration for a given vial
        :return: The calculated optical density 600 value. Same kind of OD you would see on a spectrometer or plate reader.
        """
        x = data[0]  # OD 135 data
        y = data[1]  # OD 90 data
        z = float(c0 + c1 * y + c2 * x + c3 * y ** 2 + c4 * x * y + c5 * x ** 2)
        return z

    def raw_zeroing(self,window_size, vial_num, rawdf90, rawdf135):
        """
        Calculates what raw OD 135 & OD 90 values that correspond to an optical density of zero. Then, adjusts the
        raw OD's by their respective zeroing factors, and calculates an optical density based off of this.

        :param window_size: Number of readings used when taking the median to find baseline values of OD 135, 90 when
        the culture has nothing growing in them. 20-30 should suffice.
        :param vial_num: The vial number. Integer from 0 to 15.
        :return:
        """
        #Getting the 3D calibration parameters for a specific vial:
        c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{vial_num}')

        #Figuring out which raw OD135, OD90 values give 0 based on the 3d calibration function
        #We have to used optmization for this, as we are doing the funciton "backwards"
        b135 = (49000, 57000)  #bounds for the raw OD 135 values
        b90 = (25000,30000)     #bounds for the raw OD 90 values
        bnds = (b135, b90)

        #Using minimization to find a set of raw OD135, OD 90 values that give an optical density close to zero.
        # We could probably speed up this calculation...
        res = minimize(self.opt_minimize,x0=np.array([53000,25250]),args = ([c0, c1, c2, c3, c4, c5]), bounds = bnds)

        #Getting the median of raw od readings over a window...
        od_90_baseline_median = np.median(rawdf90["OD"][:window_size])
        od_135_baseline_median = np.median(rawdf135["OD"][:window_size])

        factor = [float(res.x[0]-od_135_baseline_median),float(res.x[1]-od_90_baseline_median)]
        # Adding on the raw adjustment factor:
        # df_135 = pd.DataFrame()
        # df_90 = pd.DataFrame()
        # df_135["OD"] = copy.deepcopy(rw_df_135["OD"]) + float(factor[0])
        # df_90["OD"] = copy.deepcopy(rw_df_90["OD"]) + float(factor[1])
        od_zero_with_raw = np.real([self.three_dim([float(x), float(y)], c0, c1, c2, c3, c4, c5) for x, y in
                                    zip(rawdf135["OD"] + float(factor[0]), rawdf90["OD"] + float(factor[1]))])

        return od_zero_with_raw

    def od_zeroing(self,vial_num, df_135, df_90,window_size = 60):
        """

        Subtracts the baseline optical densities from the calculated optical densities.

        :param vial_num: the vial number, integer from 0-15
        :param df_135: a dataframe created by the getrawdf function that contains time, and raw OD 135 values
        :param df_90: Same as df_135, must be the same length, contains raw OD 90 values instead
        :param window_size: The number of baseline readings at the beggining that should be used to "zero"
        future OD readings. Readings are taken every 20 seconds, (i.e window size of 3 takes the median over 60 seconds
        and substracts this from all readings in the dataset.
        :return: optical density with the baseline OD subtracted.
        """
        i = vial_num
        #Getting the actual OD readings:
        c0, c1, c2, c3, c4, c5 = cal_3d_params.get(f'Vial{i}')
        od = np.real([self.three_dim([float(x), float(y)], c0, c1, c2, c3, c4, c5) for x, y in
                      zip(df_135["OD"], df_90["OD"])])

        #Getting the average OD readings over the window range:
        baseline_median = np.median(od[:window_size])
        adj_factor = 0- baseline_median #Getting adjustment factor
        adj_od = od + adj_factor #adjusting the optical densities accordingly.
        return adj_od

    def plot_OD_Data(self):
        """
        Creates plots of optical density versus time. The "bokeh" plots will be launched on a web browser as an html
         and have an interactive legend + zoom function.
        There are 4 bokeh panels: zeroed with raw OD values, zeroed with calculated OD values, OD 135 raw, OD 90 raw.

        The "matplotlib"  plot can be used if you need a picture for a presentation.
        It's easier to edit.

        """
        #Making an array of colours
        colour_array = ['black', 'rosybrown', 'maroon', 'salmon', 'peru', 'goldenrod', 'olive', 'lawngreen', 'forestgreen',
                        'aquamarine', 'cyan',
                        'deepskyblue', 'grey', 'blue', 'violet', 'magenta']
        # Assigning a name based on vial numbers:
        os_dict = {0: '220 +', 1: '220 -', 2: '220 +', 3: '220 +', 4: '', 5: '455+', 6: '455+', 7: 'Sterile', 8: '455+',
                   9: '455-', 10: '', 11: '925 +', 12: '925 +', 13: '925 +', 14: '', 15: '925 -'}

        #Initializing bokeh plot:
        b1 = figure(title="OD zeroed with raw", x_axis_label='Time (hours)', y_axis_label='OD')
        b2 = figure(title="OD zeroed", x_axis_label='Time (hours)', y_axis_label='OD')
        b3 = figure(title = "OD 135 degrees Raw", x_axis_label = 'Time (hours)', y_axis_label = 'OD')
        b4 = figure(title = "OD 90 degrees Raw", x_axis_label = 'Time (hours)', y_axis_label = 'OD')

        #Initializing matplotlib plot:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

        #Looping through each of the vials & making plots for them!
        for i in range(0,16):

            #Making pandas datafames out of the raw OD 135, OD 90 readings:
            raw_df_90 = self.get_raw_df(self.od_90_files[i])
            raw_df_135 = self.get_raw_df(self.od_135_files[i])

            #Making sure these are the same length. Sometimes the readings might come a bit offset from eachother
            #as they are saving from the eVOLVER

            if len(raw_df_90)>len(raw_df_135): #if 90 is longer than 135, chop 90
                raw_df_90 = raw_df_90[:len(raw_df_135)]

            elif len(raw_df_135)>len(raw_df_90): #if 135 is longer than 90, chop 135
                raw_df_135 = raw_df_135[:len(raw_df_90)]

            #Getting the time array:
            time = raw_df_90["Time"]

            #Calculating optical densities by zeroing them with the initial (n=window size) optical densities:
            od_zero_with_od = self.od_zeroing(i,raw_df_135,raw_df_90,window_size=1)

            #Calculating optical densities by zeroing the underlying raw OD 90, OD 135 values:
            od_zero_with_raw = self.raw_zeroing(window_size=100,vial_num = i,rawdf90= raw_df_90,rawdf135= raw_df_135)

            #Making 4 bokeh plots. Smoothing the OD's with a median filter.
            b1.line(time, medfilt(np.array(od_zero_with_raw),kernel_size=5),line_width=1, color = colour_array[i], legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')
            b2.line(time, medfilt(np.array(od_zero_with_od),kernel_size=5),line_width = 1, color = colour_array[i],legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')
            b3.line(raw_df_135["Time"],raw_df_135["OD"],color = colour_array[i], line_width = 1, legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}' )
            b4.line(raw_df_90["Time"], raw_df_90["OD"], color=colour_array[i], line_width = 1, legend_label=f'mOsm = {os_dict.get(i)}' + f'Vial{i}')

            # plt.plot(time, medfilt(np.array(od_zero_with_od),kernel_size=5), color = colour_array[i],label =f'Vial {i} + mOsm = {os_dict.get(i)} ')
            cut_at = int(len(time)*.25)
            m_time = time[:cut_at]
            m_od = medfilt(np.array(od_zero_with_od[:cut_at]),kernel_size=5)
            if i in [0, 2, 3]:
                ax1.plot(m_time, m_od, color=colour_array[i], label=f'Vial {i}',linewidth = .5)
                ax1.title.set_text("Low Osmolality")
                ax1.legend()
                ax1.set_ylim(0, .5)
                ax1.grid(True)
                ax1.set_ylabel("OD")

            if i in [5, 6, 8]:
                ax2.plot(m_time, m_od, color=colour_array[i], label=f'Vial {i}',linewidth = .5)
                ax2.set_ylim(0, .5)
                ax2.title.set_text("Medium Osmolality")
                ax2.legend()
                ax2.grid(True)
                ax2.set_ylabel("OD")
            if i in [11, 12, 13]:
                ax3.plot(m_time, m_od, color=colour_array[i], label=f'Vial {i}',linewidth = .5)
                ax3.title.set_text("High Osmolality")
                ax3.legend()
                ax3.set_ylim(0, .5)
                ax3.grid(True)
                ax3.set_ylabel("OD")


        #Adding a interactive legend for Bokeh plots:
        for b in [b1, b2,b3,b4]:
            b.legend.location = "top_left"
            b.legend.click_policy = "hide"

        #Showing the Bokeh plot:
        grid = gridplot([b1, b2, b3,b4], ncols=1, sizing_mode = 'stretch_width')
        output_file("od_plots.html")
        show(grid)

        #Annotating the matplotlib
        #plt.title("Full-scale eVOLVER trial")
        plt.xlabel('Time (hours)')
        # plt.ylabel('OD')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':

    # cal_3d_params = np.load(r'C:\Users\eric1\PycharmProjects\dpu\Eric_Graphing\Eric_Apr24_20223dcal.npy', allow_pickle='TRUE').item()
    #
    # od_90_folder = r'C:\Users\eric1\PycharmProjects\dpu\experiment\template\Apr24_2022_tst_expt\od_90_raw'
    # od_135_folder =r'C:\Users\eric1\PycharmProjects\dpu\experiment\template\Apr24_2022_tst_expt\od_135_raw'

    cal_3d_params = np.load(r'C:\Users\eric1\PycharmProjects\dpu\Eric_Graphing\Feb43DCal.npy',
                            allow_pickle='TRUE').item()

    od_90_folder = r'C:\Users\eric1\PycharmProjects\dpu\experiment\template\April_28_Phage_Osmo_expt\od_90_raw'
    od_135_folder = r'C:\Users\eric1\PycharmProjects\dpu\experiment\template\April_28_Phage_Osmo_expt\od_135_raw'

    OD_Data = Zeroing(cal_3d_params,od_90_folder=od_90_folder,od_135_folder=od_135_folder)
    OD_Data.plot_OD_Data()

