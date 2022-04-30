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
# from bokeh.layouts import row

def sigmoid(x, a, b, c, d):
    return np.real(a + (b - a)/(1 + (10**((c-x)*d))))

def inv_sigmoid(x, a,b,c,d):
    od_coefficients = [a,b,c,d]
    val = np.real(od_coefficients[2] - ((np.log10((od_coefficients[1] - od_coefficients[0]) /
                       (float(x) -
                        od_coefficients[0]) - 1)) /
             od_coefficients[3]))
    return val

def objective(x,y,a,b,c,d):
    return abs(y-sigmoid(x,a,b,c,d))

def third_order(x,a,b,c,d):
    return b*(x**2) + c*x + d
    return a*(x**3) + b*(x**2) + c*x + d

def three_dim(data, c0, c1, c2, c3, c4, c5):
    x = data[0]   # OD 135 data
    y = data[1]   # OD 90 data
    z= float(c0 + c1*y + c2*x + c3*y**2 + c4*x*y + c5*x**2)
    return z

def run2Dcal(filepath):

    cal_dict = np.load(filepath, allow_pickle='TRUE ').item()
    medians = cal_dict["medians"]

    standard_deviations = cal_dict["standard_deviations"]
    measured_data = cal_dict["measured_data"]

    #Initializing a dictionary which will save curve-fitting parameters
    calibration_params= dict()
    r_squared_vals = dict()



    #Setting up graphing grid:
    fig, ax = plt.subplots(4, 4)
    if 'OD90' in filepath:
        fig.suptitle("OD 90 Calibration")
    if 'OD135' in filepath:
        fig.suptitle("OD 135 Calibration")

    plt.subplots_adjust(hspace = 0.8)


    for i,vial_OD in enumerate(measured_data):
        vial_resist = medians[i]
        resist_skel = list(range(int(min(vial_resist)),int(max(vial_resist)),1))

        #Running scipy curve fit to thirds order polynomial:
        params,_ = curve_fit(sigmoid,vial_OD,vial_resist,p0 = [62721, 62721, 0, -1], maxfev=1000000)

        # params,_ = curve_fit(sigmoid,vial_resist,vial_OD,maxfev=100000)
        a,b,c,d = list(params) # error = predicted_od-vial_OD
        print(f'Calibration Vial{i}')
        print("a=",a,"b=",b,"c=",c,"d=",d)

        #Saving these parameters:
        calibration_params[f'Vial{i}'] = [a,b,c,d]

        #Creating a theoretical  calibration curve
        graph_od_curve = [inv_sigmoid(x,a,b,c,d) for x in resist_skel]

        #Solving for the OD for each given resistance using the calibration curve
        poly_predicted_od = [inv_sigmoid(x,a,b,c,d) for x in vial_resist]

        #Running stats to see how represenative the calibration curve is.
        error = np.array(poly_predicted_od)-np.array(vial_OD)
        SE = np.square(error)
        MSE = np.mean(SE)
        RMSE = np.sqrt(MSE)
        Rsquared = 1-(np.var(error) / np.var(vial_OD))
        print(f'Vial Number {i}', "RMSE", RMSE, "R Squared", Rsquared)

        #Saving the Rsquared value"
        r_squared_vals[f'Vial{i}'] =Rsquared

        #Plotting the grid section :
        ax[i // 4, (i % 4)].plot(resist_skel,graph_od_curve)
        ax[i // 4, (i % 4)].scatter(vial_resist,vial_OD,s=3,color = 'orange')
        ax[i // 4, (i % 4)].set_title(f'Vial{i}',fontsize = 'small')
        ax[i // 4, (i % 4)].tick_params(axis='x', labelsize= 8)
        ax[i // 4, (i % 4)].tick_params(axis = 'y' ,labelsize= 8)




        #Plotting the calibration figure:
        # plt.figure()
        # plt.title(f'Vial {i}')
        # plt.xlabel('Resistance')
        # plt.ylabel('OD')
        # plt.plot(resist_skel,graph_od_curve)
        # plt.scatter(vial_resist,vial_OD)
        # plt.draw()

    plt.draw()
    np.array(calibration_params)

    if 'OD90' in filepath:
        np.save("OD90_Calibration_params", calibration_params)
        np.save("OD90_R_sq", r_squared_vals)
    else:
        np.save("OD135_Calibration_params", calibration_params)
        print("saving od135 r squared..")
        np.save("OD135_R_sq", r_squared_vals)

def file_num(filename):

    start_index = int(str.index(filename, "vial")+4)
    end_index = int(str.index(filename, "_OD"))
    filenumber = int(filename[start_index: end_index])
    return filenumber

def remove_offset(OD, target_baseline=30000):
    vial_baseline = np.mean(OD[:1])
    add_on = target_baseline- vial_baseline
    newOD = OD+ add_on
    return newOD

def plot_3D_data(od_90_folder, od_135_folder, datestring):

    #Getting the OD 90 files:
    od_90_files_list = []
    for path, subdirs, files in os.walk(od_90_folder):
        for name in files:
            x = os.path.join(path, name)
            od_90_files_list.append(x)

    od_90_files_list.sort(key=file_num)

    #Getting the OD 135 files:
    od_135_files_list = []
    for path, subdirs, files in os.walk(od_135_folder):
        for name in files:
            x = os.path.join(path, name)
            od_135_files_list.append(x)

    od_135_files_list.sort(key=file_num)

    #Getting the 3D calibration parameters file:
    cal_3d_params = np.load(r'C:\Users\erlyall\PycharmProjects\dpu\Eric_Graphing\Feb43DCal.npy', allow_pickle='TRUE').item()

    #Initializing the plot:

    colour_array = ['black','rosybrown','maroon', 'salmon','peru','yellow','olive','lawngreen','forestgreen','aquamarine','cyan',
                    'deepskyblue','grey','blue','violet','magenta']
    fig, (ax1, ax2) = plt.subplots(1, 2)    #Side by side plots for raw signals
    fig, ax = plt.subplots(1,1)             #Large plot of OD with 3D calibration
    fig,log_ax = plt.subplots()   #logarithmic plot

    vial_dict = dict()

    #For bokeh plot:
    bok_plot = figure(title="OD Versus Time", x_axis_label='Time (hours)', y_axis_label='OD')

    for i,(OD_90_filepath, OD_135_filepath) in enumerate(zip(od_90_files_list,od_135_files_list)):

        #Creating the current dataframe:
        df_90 = pd.read_csv(OD_90_filepath)
        df_135 = pd.read_csv(OD_135_filepath)

        #Changing the column names:
        cols_90 = list(df_90.columns)
        df_90= df_90.rename(columns={cols_90[0]:"Time", cols_90[1]:"OD"})

        cols_135 = list(df_135.columns)
        df_135 = df_135.rename(columns={cols_135[0]:"Time", cols_135[1]:"OD"})


        #Converting dataframe columsn into lists:
        time = np.array(df_90["Time"].tolist())
        raw_od_90 = np.array(df_90["OD"].tolist())
        raw_od_135 = np.array(df_135["OD"].tolist())

        #Retriving the 3D calibration curve:
        c0, c1, c2, c3, c4, c5= cal_3d_params.get(f'Vial{i}')

        #Converting raw data into OD using 3D function:
        od = np.real([three_dim([float(x),float(y)],c0,c1,c2,c3,c4,c5) for x,y in zip(raw_od_135,raw_od_90)])

        od_offset_rem = remove_offset(od,target_baseline=0) ## Zeroing all of the initial OD's
        # od_offset_rem = od
        #Adding a median filter
        od_offset_rem = medfilt(np.array(od_offset_rem), kernel_size=7)

        #Removing offset on raw data ( making it all start at the same spot...
        raw_od_90_offset_rem = raw_od_90 #remove_offset(raw_od_90)
        raw_od_135_offset_rem = raw_od_135 # remove_offset(raw_od_135)

        #Adding to vial od dict:
        vial_dict[f'Vial{i}'] = od_offset_rem

        #Plotting raw OD
        ax1.plot(time, raw_od_90_offset_rem, label= f'Vial{i}', color = colour_array[i])
        ax1.grid(True)
        ax2.plot(time, raw_od_135_offset_rem, label= f'Vial{i}', color = colour_array[i])
        ax2.grid(True)

        #Plotting converted od:
        os_dict = {0: '226', 1: '400', 2: '300', 3: '500', 5: '800', 7: '226 ', 8: '400', 9: '226', 12: '500', 13: '300',
                   15: '800'}
        os_color_dict = {'226': 'black', '300': 'blue', '400': 'cyan', '500': 'green', '600': 'yellow', '800': 'orange'}
        if i in [0,1,2,5,6,8,11,12]:
            ax.plot(time, od_offset_rem, color = colour_array[i], label = f'Vial{i}')
            # Plotting on bokeh plot:
            bok_plot.line(time, od_offset_rem,line_width=2, color = colour_array[i],legend =  f'Vial{i}')


    output_file("od_plots.html")
    show(bok_plot)


    ax1.set_title("Raw OD 90 vs Time"+ datestring)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Raw OD 90')
    ax1.legend()

    ax2.set_title("Raw OD 135 vs Time" + datestring)
    ax2.set_xlabel('Time')
    ax2.set_ylabel("Raw OD 135")
    ax2.legend()

    ax.set_title("Turbidostat Mode", fontsize = 'xx-large')
    ax.set_xlabel('Time (hrs)', fontsize = 'xx-large')
    ax.set_ylabel("OD", fontsize = 'xx-large')
    ax.legend()
    #
    # log_ax.set_title('Log Od vs Time')
    # log_ax.set_xlabel('Time (hrs)')
    # log_ax.set_ylabel('Log OD')
    # # log_ax.set_ylim(bottom = 0)
    # log_ax.legend()

    # Plotting forwards difference derivative.. might be a little too high res...
    # plot_derivatives(vial_dict,time,colour_array, datestring)

    #Fitting an exp curve to one of the datapoints:
    # timestep = np.mean (np.diff(np.array(time)))
    # exp_curve_fit(medfilt(vial_dict.get('Vial5'),21),time,st_time=.2,end_time=2,time_step=timestep)


    # #Plotting endpoint data:
    # endpt_od= [1.108,1.186,1.166,1.127,1.169,1.139,1.139,1.149,1.193,1.205,1.113,1.161,1.162,1.164,1.206,1.184,1.206]
    # endpt_time = [19.25]*16
    # for i in range(16):
    #     ax2.scatter(endpt_time[i],endpt_od[i],color = colour_array[i])

    #Plotting endpoint difference data:
    # compare_endpt_data(vial_dict,endpt_od)
    plt.show()

def central_difference(array,a,h,timestep):
    if a-h < 0 :
        #forward difference
        return (array[a+h] - array[a])/(h*timestep)
    elif a+h > len(array):
        #Backward difference
        return (array[a] - array[a-h])/(h*timestep)
    else:
        #Central Difference
        return (array[a+int(h/2)] - array[a-int(h/2)]) / (h*timestep)

def compare_endpt_data(vial_od_dict,endpt_ods):

    diff = []
    vialnames = []
    for i in range(16):
        vial_od = vial_od_dict.get(f'Vial{i}')
        end_avg = np.mean(vial_od[-5:])
        diff.append(end_avg - endpt_ods[i])
        vialnames.append(f'Vial{i}')

    plt.figure()
    plt.title("Difference between endpoint evolver and spec OD's")
    plt.ylabel("OD")
    plt.xlabel("Vial")
    plt.bar(vialnames,diff)
    plt.draw()

def plot_derivatives(od_folder,color_legend, datestring):
    #Plots a forward difference derivative.
    #Finding the timestep between data readigngs:

    fig, ax = plt.subplots()
    for i in range(0,len(od_folder)):
        df = get_raw_df(od_folder[i])
        timestep = np.mean(np.diff(np.array(df['Time'])))
        vial_od = df['OD']
        h= 50
        od_diff = [central_difference(vial_od,a,h,timestep=timestep) for a in range(0,len(vial_od))]
        od_diff = medfilt(od_diff,kernel_size=25)
        print(f'Vial {i} Max Derivative', np.max(od_diff))
        timespace = [timestep*i for i in range(0,len(od_diff))]
        ax.plot(timespace,medfilt(od_diff,kernel_size=3),color = color_legend[i], label = f'Vial{i}')
        # ax.set_ylim((-1,1))
        ax.set_xlim((7.5,20))

    ax.set_title("First derivative plot" + datestring)
    ax.set_xlabel('Time')
    ax.set_ylabel("d(OD)/dt")
    ax.legend()

    plt.draw()

def exp_growth(t,x0,u,C):
    od = np.real(C + x0*(np.exp(u*t)))
    return od

def exp_curve_fit(vial_od,time, st_time,end_time,time_step):
    #Find start, finish indexes to start trial at:
    st_index = int(st_time/time_step)
    end_index = int(end_time/time_step)

    #Creating od, time slices to fit an exp function over:
    od_slice = np.array(vial_od[st_index:end_index])
    time_slice =np.array(time[st_index:end_index])

    #Running a scipy curve fit to an exponential function:
    params, _ = curve_fit(exp_growth, time_slice, od_slice, p0=[.01,.8,0], maxfev=1000000)

    x0,u,C = list(params)

    #Fitting a theoretical curve based  on params:
    theory_cell_conc = [exp_growth(float(t),x0,u,C) for t in time_slice]

    #plotting data
    fig,ax = plt.subplots()
    ax.set_title("OD vs time with fitted growth curve")
    ax.plot(time_slice,od_slice,color = 'blue',label = 'Original')
    ax.plot(time_slice,theory_cell_conc,color = 'Orange', label = 'Theoretical')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD')
    print("x0 = ",x0, " u= ", u, "c = ", C)
    plt.legend()
    plt.show()


def create_od_file_list(od_folder):
    #Takes in a od90 or od 135 folder, and returns all the file names in a list in order of vials.
    od_files_list = []
    for path, subdirs, files in os.walk(od_folder):
        for name in files:
            x = os.path.join(path, name)
            od_files_list.append(x)
    od_files_list.sort(key=file_num)
    return od_files_list # a list of filenames containing the raw od data, in order of the vials.

def get_raw_df(filepath):
    df= pd.read_csv(filepath)
    cols = list(df.columns)
    df = df.rename(columns = {cols[0]: "Time", cols[1]: "OD"})
    return df
if __name__ == '__main__':

    #colours:
    colour_array = ['black', 'rosybrown', 'maroon', 'salmon', 'peru', 'yellow', 'olive', 'lawngreen', 'forestgreen',
                    'aquamarine', 'cyan',
                    'deepskyblue', 'grey', 'blue', 'violet', 'magenta']

    #Getting a list of the vials:
    od_files_list = create_od_file_list(r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\Apr27_cal_tst_expt\OD')
    plot_derivatives(od_files_list,colour_array,"Testing")
    plt.show()

