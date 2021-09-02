import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import Panel, Tabs
from bokeh.models import Span



#Funciton that turns txt files into pandas dataframes.
def get_raw_df(filepath,name):
    df = pd.read_csv(filepath)
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "Time", cols[1]: name})
    return df



#Getting the experiment names:
EXP_NAME = 'July_9_negative_cntrl_expt'
save_path = r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template'

#Turning the pump calibration file into a numpy array?
file_path = (r'C:\Users\erlyall\PycharmProjects\dpu\experiment\template\pump_cal.txt')
flow_calibration = np.loadtxt(file_path, delimiter="\t")[0]


#Getting colours and legends:
os_dict = {0: ' v0 220 +', 1: 'v1 220 -', 2: 'v2 220 +', 3: 'v3 220 +', 4: 'v4', 5: ' v5 455+', 6: 'v6 455+', 7: 'v7 Sterile', 8: 'v8 455+',
           9: 'v9 455-', 10: 'v10', 11: 'v11 925 +', 12: 'v12 925 +', 13: 'v13 925 +', 14: 'v14', 15: 'v15 925 -'}

colour_array = ['black', 'rosybrown', 'maroon', 'salmon', 'peru', 'yellow', 'olive', 'lawngreen', 'forestgreen',
                        'aquamarine', 'cyan', 'deepskyblue', 'grey', 'blue', 'violet', 'magenta']

#Initializing the bokeh plots:

gr_plot = figure(title="Growth Rate Plots", x_axis_label='Time (hrs)', y_axis_label='Growth Rate h-1', sizing_mode = 'stretch_width')
# p = figure(x_range=fruits, plot_height=250, title="Fruit counts",
#            toolbar_location=None, tools="")
dil_plot = figure(title = "Dilution Plots", x_range = [f'{os_dict.get(i)}' for i in range(0,16)], sizing_mode = 'stretch_width')
temp_plot = figure(title = "Temperature Plots", x_axis_label = 'Time (hrs)', y_axis_label = 'Temp (c)', sizing_mode = 'stretch_width')
dil_time_plot = figure(title="Media vs time",x_axis_label = 'Time  (hrs)', y_axis_label = 'Media consumed (L)', sizing_mode = 'stretch_both')




# Creating an arrayto save diliution volumes:
all_vial_consumptions = []

#Running a bokeh plot on all the vials:
for x in range(0,16):

    #Plotting growth rate data:
    file_name = "vial{0}_gr.txt".format(x)
    file_path = os.path.join(save_path, EXP_NAME,'growthrate', file_name)
    data = get_raw_df(file_path, name = "GR")
    time = data['Time'].tolist()
    gr = data['GR'].tolist()
    gr_plot.line(time, gr, line_width=1, color=colour_array[x],
            legend_label=f'mOsm = {os_dict.get(x)}')

    #Plotting temperature data:
    file_name = "vial{0}_temp.txt".format(x)
    file_path = os.path.join(save_path, EXP_NAME, 'temp', file_name)
    data = get_raw_df(file_path, name="temp")
    time = data['Time'].tolist()
    temp = data['temp'].tolist()
    temp_plot.line(time, temp, line_width=1, color=colour_array[x],
                 legend_label=f'mOsm = {os_dict.get(x)}')

    #Retrieving diltuion data:
    file_name = "vial{0}_pump_log.txt".format(x)
    file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)
    data = get_raw_df(file_path, name="dil")
    dil_times = data['dil'].tolist()[1:]
    dil_vols =[flow_calibration[x] *1.07 *  el for el in dil_times]
    volume_progression = [sum(dil_vols[:i]) for i in range(0,len(dil_vols))]
    print("Volume progression length", len(volume_progression))
    print("Time length", len(data['Time'].tolist()[1:]))

    dil_time_plot.line(data['Time'].tolist()[1:],volume_progression,line_width=1, color=colour_array[x],
                 legend_label=f'mOsm = {os_dict.get(x)}')
    total_dil_vol = np.sum(np.array(dil_vols))
    all_vial_consumptions.append(total_dil_vol)



#Plotting a barchart of the total media consumption:
dil_plot.vbar(x=[f'{os_dict.get(i)}' for i in range(0,16)], top=all_vial_consumptions,width = 0.9)

#Plotting horizontal line on the dil time plot for max volume:
hline = Span(location=1800, dimension='width', line_color='green', line_width=3)
dil_time_plot.add_layout(hline)


# Outputting the bokeh plot:
for p in [gr_plot,temp_plot,dil_time_plot]:
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

#Creating the name of the html file to output:
output_file("Growth_Temp_Media_Plots.html")

#plottin this as a series of tabs for each type of plot:
tab1 = Panel(child=gr_plot, title="Growth Rates")
tab2 = Panel(child = temp_plot, title = "Temperature")
tab3 = Panel(child = dil_plot,title = "Total Media Consumption")
tab4 = Panel(child = dil_time_plot,title = 'Rate of media consumption')

show(Tabs(tabs=[tab1, tab2, tab3,tab4]))
