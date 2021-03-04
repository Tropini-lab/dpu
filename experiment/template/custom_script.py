#!/usr/bin/env python3

import numpy as np
import logging
import os.path
import time

# logger setup
logger = logging.getLogger(__name__)

##### USER DEFINED GENERAL SETTINGS #####

#set new name for each experiment, otherwise files will be overwritten
EXP_NAME = 'Mar_2_Turb_expt'
EVOLVER_IP = '10.0.0.100'
EVOLVER_PORT = 8081

##### Identify pump calibration files, define initial values for temperature, stirring, volume, power settings

TEMP_INITIAL = [37] * 16 #degrees C, makes 16-value list
#Alternatively enter 16-value list to set different values
#TEMP_INITIAL = [30,30,30,30,32,32,32,32,34,34,34,34,36,36,36,36]

STIR_INITIAL = [10] * 16 #try 8,10,12 etc; makes 16-value list
#Alternatively enter 16-value list to set different values
#STIR_INITIAL = [7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10]

VOLUME =  26 #mL, determined by vial cap straw length
PUMP_CAL_FILE = 'pump_cal.txt' #tab delimited, mL/s with 16 influx pumps on first row, etc.
OPERATION_MODE = 'turbidostat' #use to choose between 'turbidostat' , 'chemostat' and 'eric_chemostat' functions
# if using a different mode, name your function as the OPERATION_MODE variable

##### END OF USER DEFINED GENERAL SETTINGS #####

def turbidostat(eVOLVER, input_data, vials, elapsed_time):
    OD_data = input_data['transformed']['od']

    ##### USER DEFINED VARIABLES #####

    turbidostat_vials = [5] #vials is all 16, can set to different range (ex. [0,1,2,3]) to only trigger tstat on those vials
    stop_after_n_curves = np.inf #set to np.inf to never stop, or integer value to stop diluting after certain number of growth curves
    OD_values_to_average = 5  # Number of values to calculate the OD average

    # lower_thresh = [0.2] * len(vials) #to set all vials to the same value, creates 16-value list
    # upper_thresh = [0.4] * len(vials) #to set all vials to the same value, creates 16-value list

    #Alternatively, use 16 value list to set different thresholds, use 9999 for vials not being used
    lower_thresh = [0.1, 0.2, .15, 99, 9999, .15, 9999, 9999, 9999, .1, .2, 9999, 9999, 9999, 9999, 9999]
    upper_thresh = [0.2, 0.25, .2, 99, 9999, .2, 9999, 9999, 9999, .2, .25, 9999, 9999, 9999, 9999, 9999]


    ##### END OF USER DEFINED VARIABLES #####


    ##### Turbidostat Settings #####
    #Tunable settings for overflow protection, pump scheduling etc. Unlikely to change between expts

    time_out = 5 #(sec) additional amount of time to run efflux pump
    pump_wait = 2.5 # (min) minimum amount of time to wait between pump events

    ##### End of Turbidostat Settings #####

    save_path = os.path.dirname(os.path.realpath(__file__)) #save path
    flow_rate = eVOLVER.get_flow_rate() #read from calibration file


    ##### Turbidostat Control Code Below #####

    # fluidic message: initialized so that no change is sent
    MESSAGE = ['--'] * 48
    for x in turbidostat_vials: #main loop through each vial

        # Update turbidostat configuration files for each vial
        # initialize OD and find OD path

        file_name =  "vial{0}_ODset.txt".format(x)
        ODset_path = os.path.join(save_path, EXP_NAME, 'ODset', file_name)
        data = np.genfromtxt(ODset_path, delimiter=',')
        ODset = data[len(data)-1][1]
        ODsettime = data[len(data)-1][0]
        num_curves=len(data)/2;

        file_name =  "vial{0}_OD.txt".format(x)
        OD_path = os.path.join(save_path, EXP_NAME, 'OD', file_name)
        data = eVOLVER.tail_to_np(OD_path, OD_values_to_average)
        average_OD = 0

        # Determine whether turbidostat dilutions are needed
        #enough_ODdata = (len(data) > 7) #logical, checks to see if enough data points (couple minutes) for sliding window
        collecting_more_curves = (num_curves <= (stop_after_n_curves + 2)) #logical, checks to see if enough growth curves have happened

        if data.size != 0:
            # Take median to avoid outlier

            #If the last pumping event is < 5 min, reduce the median filter size to
            od_values_from_file = data[:,1]
            print("The od values from a file", od_values_from_file)
            average_OD = float(np.median(od_values_from_file))



            #if recently exceeded upper threshold, note end of growth curve in ODset, allow dilutions to occur and growthrate to be measured
            if (average_OD > upper_thresh[x]) and (ODset != lower_thresh[x]):
                text_file = open(ODset_path, "a+")
                text_file.write("{0},{1}\n".format(elapsed_time,
                                                   lower_thresh[x]))
                text_file.close()
                ODset = lower_thresh[x]
                # calculate growth rate
                eVOLVER.calc_growth_rate(x, ODsettime, elapsed_time)

            #if have approx. reached lower threshold, note start of growth curve in ODset
            if (average_OD < (lower_thresh[x] + (upper_thresh[x] - lower_thresh[x]) / 3)) and (ODset != upper_thresh[x]):
                text_file = open(ODset_path, "a+")
                text_file.write("{0},{1}\n".format(elapsed_time, upper_thresh[x]))
                text_file.close()
                ODset = upper_thresh[x]


            #if need to dilute to lower threshold, then calculate amount of time to pump

            # if average_OD > ODset and collecting_more_curves: #This line can result in double pumps for a single diltuion
            if average_OD > upper_thresh[x]:
                print("Average OD ",average_OD, " past threshold ", upper_thresh[x])

                time_in = - (np.log(lower_thresh[x]/average_OD)*VOLUME)/flow_rate[x]

                #TODO: Can influx pumps run longer than 20 seconds? Test this.

                if time_in > 37:  #Eric changed this threshold from 20 to 35 seconds.. This way pumping events don't have to be split up.
                    time_in = 37

                time_in = round(time_in, 2)

                file_name =  "vial{0}_pump_log.txt".format(x)
                file_path = os.path.join(save_path, EXP_NAME,
                                         'pump_log', file_name)
                data = np.genfromtxt(file_path, delimiter=',')
                last_pump = data[len(data)-1][0]

                if ((elapsed_time - last_pump)*60) >= pump_wait: # if sufficient time since last pump, send command to Arduino. Eric upped this from 60s
                    print("Time since last pump = ", (elapsed_time-last_pump)*60, "triggering dilution" )
                    logger.info('turbidostat dilution for vial %d' % x)
                    # influx pump
                    MESSAGE[x] = str(time_in)
                    # efflux pump
                    MESSAGE[x + 16] = str(time_in + time_out)

                    file_name =  "vial{0}_pump_log.txt".format(x)
                    file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)

                    text_file = open(file_path, "a+")
                    text_file.write("{0},{1}\n".format(elapsed_time, time_in))
                    text_file.close()
        else:
            logger.debug('not enough OD measurements for vial %d' % x)

    # send fluidic command only if we are actually turning on any of the pumps
    if MESSAGE != ['--'] * 48:
        #Run every efflux pump each time any influx pump runs
        # This protects the system against overflows in case vials are connected wrongly....

        #Find the max efflux pumping time in the MESSAGE
        num_message = [int(x) for x in MESSAGE if x != '--']
        max_time = max(num_message)

        #Assign all efflux pumps to the max pumping time.
        for index in range(16,32):
            MESSAGE[index] = str(max_time)

        #Sending fluid commands to evolver:
        eVOLVER.fluid_command(MESSAGE)

        # your_FB_function_here() #good spot to call feedback functions for dynamic temperature, stirring, etc for ind. vials
    # your_function_here() #good spot to call non-feedback functions for dynamic temperature, stirring, etc.

    # end of turbidostat() fxn

def chemostat(eVOLVER, input_data, vials, elapsed_time):
    OD_data = input_data['transformed']['od']

    ##### USER DEFINED VARIABLES #####
    start_OD = .2 # ~OD600, set to 0 to start chemostat dilutions at any positive OD
    start_time = 0 #hours, set 0 to start immediately
    # Note that script uses AND logic, so both start time and start OD must be surpassed

    OD_values_to_average = 6  # Number of values to calculate the OD average
    chemostat_vials = vials #vials is all 16, can set to different range (ex. [0,1,2,3]) to only trigger tstat on those vials

    # rate_config = [1.1538] * 16 #to set all vials to the same value, creates 16-value list
    #UNITS of 1/hr, NOT mL/hr, rate = flowrate/volume, so dilution rate ~ growth rate, set to 0 for unused vials

    #Alternatively, use 16 value list to set different rates, use 0 for vials not being used
    #Only running vials 3,5,10,14
    rate_config = [0.0,0.0,0.0,0.5,0.0,1.0,0.0,0.0,0.0,0.0,1.5,0.0,0.0,0.0,2.0,0.0]


    ##### END OF USER DEFINED VARIABLES #####


    ##### Chemostat Settings #####

    #Tunable settings for bolus, etc. Unlikely to change between expts
    bolus = 0.5 #mL, can be changed with great caution, 0.2 is absolute minimum

    ##### End of Chemostat Settings #####

    save_path = os.path.dirname(os.path.realpath(__file__)) #save path
    flow_rate = eVOLVER.get_flow_rate() #read from calibration file
    period_config = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #initialize array
    bolus_in_s = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #initialize array


    ##### Chemostat Control Code Below #####

    for x in chemostat_vials: #main loop through each vial

        # Update chemostat configuration files for each vial

        #initialize OD and find OD path
        file_name =  "vial{0}_OD.txt".format(x)
        OD_path = os.path.join(save_path, EXP_NAME, 'OD', file_name)
        data = eVOLVER.tail_to_np(OD_path, OD_values_to_average)
        average_OD = 0
        #enough_ODdata = (len(data) > 7) #logical, checks to see if enough data points (couple minutes) for sliding window

        if data.size != 0: #waits for seven OD measurements (couple minutes) for sliding window

            #calculate median OD
            od_values_from_file = data[:,1]
            average_OD = float(np.median(od_values_from_file))

            # set chemostat config path and pull current state from file
            file_name =  "vial{0}_chemo_config.txt".format(x)
            chemoconfig_path = os.path.join(save_path, EXP_NAME,
                                            'chemo_config', file_name)
            chemo_config = np.genfromtxt(chemoconfig_path, delimiter=',')
            last_chemoset = chemo_config[len(chemo_config)-1][0] #should t=0 initially, changes each time a new command is written to file
            last_chemophase = chemo_config[len(chemo_config)-1][1] #should be zero initially, changes each time a new command is written to file
            last_chemorate = chemo_config[len(chemo_config)-1][2] #should be 0 initially, then period in seconds after new commands are sent

            # once start time has passed and culture hits start OD, if no command has been written, write new chemostat command to file

            if ((elapsed_time > start_time) & (average_OD > start_OD)):

                #calculate time needed to pump bolus for each pump
                bolus_in_s[x] = bolus/flow_rate[x]

                # calculate the period (i.e. frequency of dilution events) based on user specified growth rate and bolus size
                if rate_config[x] > 0:
                    period_config[x] = (3600*bolus)/((rate_config[x])*VOLUME) #scale dilution rate by bolus size and volume
                else: # if no dilutions needed, then just loops with no dilutions
                    period_config[x] = 0

                if  (last_chemorate != period_config[x]):
                    print('Chemostat updated in vial {0}'.format(x))
                    logger.info('chemostat initiated for vial %d, period %.2f'
                                % (x, period_config[x]))
                    # writes command to chemo_config file, for storage
                    text_file = open(chemoconfig_path, "a+")
                    text_file.write("{0},{1},{2}\n".format(elapsed_time,
                                                           (last_chemophase+1),
                                                           period_config[x])) #note that this changes chemophase
                    text_file.close()
        else:
            logger.debug('not enough OD measurements for vial %d' % x)

        # your_FB_function_here() #good spot to call feedback functions for dynamic temperature, stirring, etc for ind. vials
    # your_function_here() #good spot to call non-feedback functions for dynamic temperature, stirring, etc.

    eVOLVER.update_chemo(input_data, chemostat_vials, bolus_in_s, period_config,immediate=True) #compares computed chemostat config to the remote one
    # end of chemostat() fxn

# def your_function_here(): # good spot to define modular functions for dynamics or feedback

def remove_baseline(raw_od_90,raw_od_135, window_size,vial_number):
    """

    @param od: an array consisting of optical densities
    @param window_size: a number of readings to establish a baseline median from
    @param vial_number: The vial number we are looking at.
    """

    #Using the large window size to make a determine a baseline OD for a given vial
    baseline_od_90 = np.median(raw_od_90[:window_size])
    baseline_od_135 = np.median(raw_od_135[:window_size])

    #Find the raw od 90 and od 135 values where the vial OD calibrates to zero:
    #Should this be done off of the 3D or 3D calibrations?? A scipy.optimize.minimize with a given range, compare this to calibration zeros.

def three_dim(data, c0, c1, c2, c3, c4, c5):
    x = data[0]
    y = data[1]
    z= float(c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2)
    return z




def combostat(eVOLVER, input_data,vials,elapsed_time):
    #Function that allows some vials to be in chemostat mode, and other vials to be in turbidostat mode..
    yeet = 1

def eric_chemostat(eVOLVER,input_data,vials,elapsed_time):

    ##USER DEFINED VARIABLES
    start_od = .2        #OD units: Vials need to reach this OD in order to start chemostat.
    start_time = 0.3      #Hours: The experiment needs to be running at least this long in order to start the chemostat
    bolus = .5          #mL : The amount of fluid to bolus each time the period is up. Make no less than 0.2
    dilution_rate = .25    #1/h, the dillution rate will equal the growth rate of the culture at steady state.
    volume = 26         #Volume media in the smart vial. This is determined by the length of the efflux straw.
    od_values_to_median = 5    # Takes the median of the past n OD readings to determine is start od threshold is reached.
    chemostat_vials = vials     # Could only select specific vials if wanted to trigger chemostat on only select vials.
    extra_pumping_time = 2      #extra pumping time for efflux pump in seconds

    #Calculated Variables:
    flowrate = eVOLVER.get_flow_rate()   # the flowrate for each pump from the calibration file in mL/second
    bolus_time = [round(bolus/float(flowrate[i]),2) for i in range(0,len(flowrate))]  #The amount of time in seconds each pump needs to run for to reach the desired bolus volume.
    pumping_period = 1/(dilution_rate * volume / bolus)   # Hours: how often, in hours, each pump needs to pump- this should be the same for all pumps!
    print("The pumping period is: ",np.mean(bolus_time),"seconds every",pumping_period,"hours")
    #Initializing a blank pumping command:
    MESSAGE = ['--'] * 48

    #Getting the base filepath:
    save_path = os.path.dirname(os.path.realpath(__file__))

    # Iterating through each pump, finding which ones a) meet starting od, b) meet starting time and c) are scheduled for a dilution.
    for x in chemostat_vials: #main loop through each vial

        od_thresh_passed = get_median_od(eVOLVER,x,save_path,od_values_to_median, start_od)
        file_name = "vial{0}_pump_log.txt".format(x)
        file_path = os.path.join(save_path, EXP_NAME, 'pump_log', file_name)

        if od_thresh_passed and elapsed_time>start_time and check_period(file_path,elapsed_time,pumping_period):
            # print(f'Conditions passed for vial {x} !')
            #Edit the pumping message
            MESSAGE[x] = str(round(bolus_time[x],2))
            MESSAGE[x+16] = str(round(bolus_time[x] + extra_pumping_time,2))

            #log that a pumping event has occured
            text_file = open(file_path, "a+")
            text_file.write("{0},{1}\n".format(elapsed_time, bolus_time[x]))
            text_file.close()
        else:
            yeet=1
            # print(f'Contiontions failed for vial {x}')

    #Send the pumping command only if pumps are pumping:
    if MESSAGE != ['--'] * 48:
        # print("Sending Fluid Command!")
        eVOLVER.fluid_command(MESSAGE)
    else:
        print("Conditions not passed, fluid commands will not send yet...")

def get_median_od(eVOLVER, x,save_path,od_values_to_median, start_od):
    # Getting the OD path
    file_name = "vial{0}_OD.txt".format(x)
    OD_path = os.path.join(save_path, EXP_NAME, 'OD', file_name)
    data = eVOLVER.tail_to_np(OD_path, od_values_to_median)

    #If there is too little data, return false
    if len(data) < od_values_to_median :
        # print("Not enough OD data! vial",x)
        return False

    #If there is enough data, calculate the median od
    od_values_from_file = data[:, 1]
    median_od = float(np.median(od_values_from_file))
    if median_od > start_od:
        # print("Od pased threshold vial",x)
        return True
    else:
        # print("Od too low vial",x)
        return False

def check_period(file_path, elapsed_time, pumping_period):
    # Get the last timepoint:
    data = np.genfromtxt(file_path, delimiter=',')
    last_pump_time = data[len(data) - 1][0]
    last_pump_length = data[len(data)-1][1]
    if len(data) <= 2:      #If there's no prior pumping events, we start one
        # print("No pumps yet,starting to pump!")
        return True
    elif (elapsed_time-float(last_pump_time)) > pumping_period and float(last_pump_length) > 0:
        # print("Reached pumping period, check condition passed!")
        return True
    else:
        # print("Have not reached pumping period, check failed.")
        return False



if __name__ == '__main__':
    print('Please run eVOLVER.py instead')
