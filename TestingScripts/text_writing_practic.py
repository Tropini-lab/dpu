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
from bokeh.layouts import row

time_in = - (np.log(.2 / .4) * 26) / .77

past_od = [0.40073911961661657,0.4009379252525592,0.40074687896078276,0.345506898142772,0.2421577268523809,0.24528354148863407]
med_od = np.median(past_od)
print("Median od for second pump", med_od)
time_in = - (np.log(.2 / med_od) * 26) / .77

print("Time in for second pump")
print(time_in)