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


def three_dim(data, c0, c1, c2, c3, c4, c5):
    x = data[0]
    y = data[1]
    z= float(c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2)
    return z

