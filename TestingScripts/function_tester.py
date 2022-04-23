import numpy as np
import pandas as pd
import os
import scipy.stats as stats

x = np.array([1,2,3,4,5,6,7,8,9])
bins = np.array([0,3,6])
inds = np.digitize(x, bins)
print(inds)

for bin_idx in set(inds):
    bin_arr = x[inds==bin_idx]
    print(bin_arr)
    print (bin_idx, np.mean(bin_arr), np.median(bin_arr))