from nptdms import TdmsFile
from sklearn import preprocessing 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

def SW_seg(df_power, width):
    # sliding window method
	'''
    Important step is to segmentation the time series data
    we use the sliding window method
    '''
    df_seg = []
    size_df = df_power.size
    anchor = 1 
    while((anchor+width*3) < size_df): 
        mid = df_power[anchor+width: anchor+width*3].idxmax()
        seg = df_power[mid-width: mid+width]
        df_seg.append(seg.values)
        anchor = mid
    return df_seg

def Trend_Change(line, delta): 
    delta_new = abs(np.diff(line))
    max_d = np.max(delta_new)
    if max_d > delta:
        delta = delta_new
        k = np.where(delta_new > max_d/2)[0][0]+1
        return k
    else: return -1
    
def Trend_Detection(s_fr, delta, win_size):
    len_s = len(s_fr)
    df = pd.Series(s_fr)
    i = 0
    j = i;
    
    while (i<len_s):
        k = Trend_Change(s_fr[j:i+win_size],delta)
        if (k != -1):
           s_fr[j:j+k].plot()
           s_fr[j+k+1:i+win_size].plot()
           j = i+win_size 
        i += win_size
    plt.figure()
