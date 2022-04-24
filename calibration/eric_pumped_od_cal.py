import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def eric_pumped_cal_od_sorting(EXP_NAME, save_path,target_ods):

    # Getting a list of pumping times
    file_name = "vial{0}_pump_log.txt".format(0) #we pumped all the vials at the same time, so we can use vial 0 as a representative
    file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)
    pump_data = np.genfromtxt(file_path, delimiter=',')

    # Truncating this to remove NA values from the first line:
    pumping_times = [pair[0] for pair in pump_data][1:]

    #Making empty lists to place OD90, OD135, and measured values in
    #This is the format the three_dimension_fit function wants things in
    od_90_datas = []
    od_135_datas = []
    measured_datas =[]

    #Getting median values of OD90, OD 130 for each vial.
    for i in range(16):
        print("Doing vial ",i)

        #Retrieving OD 90 data:
        od_90_file_name = "vial{0}_od_90_raw.txt".format(i)
        od_90 = np.genfromtxt(os.path.join(save_path, EXP_NAME,'od_90_raw', od_90_file_name), delimiter=',')
        od_90 = od_90[1:len(od_90)-1] #the first line is Na, the last line is a weird space.. get rid of them!

        #Retrieving OD 135 data:
        od_135_file_name = "vial{0}_od_135_raw.txt".format(i)
        od_135 = np.genfromtxt(os.path.join(save_path, EXP_NAME,'od_135_raw', od_135_file_name), delimiter=',')
        od_135 = od_135[1:len(od_135)-1]

        #Getting a list of the times optical densities were read at (these are the same for OD90, 135)
        od_times = np.array([p[0] for p in od_90]) #Will use the OD 90 times a a representative
        od_90_vals = np.array([p[1] for p in od_90]) #Array of OD's
        od_135_vals = np.array([p[1] for p in od_135])


        #Grouping the OD readings based on their time, and whether they were before / after a pump:
        inds = np.digitize(od_times, pumping_times) #Grouping each bin based on what pumps they were before:

        median_od_90 = []
        median_od_135 = []
        for bin_idx in set(inds):
            median_od_90.append(np.median(od_90_vals[inds == bin_idx]))
            median_od_135.append(np.median(od_135_vals[inds == bin_idx]))

        #appending all of this to our master lists.
        od_90_datas.append(median_od_90)
        od_135_datas.append(median_od_135)
        measured_datas.append(target_ods)

    return od_90_datas,od_135_datas,measured_datas

def eric_three_dimension_fit(x_datas,y_datas,z_datas, fit_name, params, graph = True):
    initial_parameters = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    coefficients = []
    datas = []

    for i in range(16):
        x_data = np.array(x_datas[i])
        y_data = np.array(y_datas[i])
        z_data = np.array(z_datas[i])

        data = [x_data, y_data, z_data]

        fitted_parameters, pcov = scipy.optimize.curve_fit(three_dim, [x_data, y_data], z_data, p0 = initial_parameters)

        modelPredictions = three_dim(data, *fitted_parameters)
        absError = modelPredictions - z_data
        SE = np.square(absError) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(z_data))
        print('Vial ' + str(i))
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)
        print('fitted prameters', fitted_parameters)

        coefficients.append(fitted_parameters.tolist())
        datas.append(data)

    if graph:
        graph_3d_data(three_dim, datas, coefficients, fit_name)

    return create_fit(coefficients, fit_name, '3d', time.time(), params)

def graph_3d_data(func, datas, coefficients, fit_name):
    fig = plt.figure()
    fig.suptitle("Fit Name: " + fit_name)
    for i, data in enumerate(datas):
        x_data = data[0]
        y_data = data[1]
        z_data = data[2]
        x_space = np.linspace(min(x_data), max(x_data), 20)
        y_space= np.linspace(min(y_data), max(y_data), 20)
        X, Y = np.meshgrid(x_space, y_space)
        Z = func(np.array([X, Y]), *coefficients[i])

        ax = fig.add_subplot(4, 4, i + 1, projection = '3d')

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, alpha=0.5)

        ax.scatter(x_data, y_data, z_data, c='r', s=10) # show data along with plotted surface

        ax.set_xlabel('OD90') # X axis data label
        ax.set_ylabel('OD135') # Y axis data label
        ax.set_zlabel('OD Measured') # Z axis data label

    plt.show()

def three_dim(data, c0, c1, c2, c3, c4, c5):
    x = data[0]
    y = data[1]
    return c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2

def create_fit(coefficients, fit_name, fit_type, time_fit, params):
    return {"name": fit_name, "coefficients": coefficients, "type": fit_type, "timeFit": time_fit, "active": False, "params": params}


if __name__ == '__main__':

    EXP_NAME = 'T3_Aug_19_pumped_od_cal_expt'
    save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'   #Need to change "eric1" if using a different computer

    # Getting a list of the target optical densities, where 1 is the starting OD (before the first pump, and 0.05
    #is the finishing OD after the last pump.
    target_ods = [1, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]

    x,y,z = eric_pumped_cal_od_sorting(EXP_NAME,save_path,target_ods)
    save_fit = eric_three_dimension_fit(x,y,z,fit_name='Eric_Pumped_cal3d',params ='eric cal, no param',graph = True)
