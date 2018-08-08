from nptdms import TdmsFile
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.mlab import PCA
import npfeintool as npF
from npfeintool import filterVar
import matplotlib.pyplot as plt


# load in data, take the TDMS data type as example
tdms_file = TdmsFile("AKF-FW1-1-H1021-1500.tdms")
#preprocessing the data type 'DINT' 
df = tdms_file.object('DINT').as_dataframe()

'''
Step 1: kick out the "all zero", "all one" value
this Step is essential for selection meaningful selection

'''
x =  npF.kill_all_x(0, df)
x = npF.keep_x_var(0.8, x)
#findout the selected columns
ix = np.isin(df.values[0,:], x[0])
column_indices = np.where(ix==True)
df = df.iloc[:,column_indices[0]]
'''
Step 2: divide the periodical time series data into unit
e.g. we select only one channel data to segementation
    Params are: dataset, windsize(the approx unit size)
'''

df_power = df['SupplyUnit 50U4 (Unidrive M701 Regen).Transmit PDO Mapping 8.Leistung']
df_seg = npF.SW_seg(df_power, 968)

'''
Step 3: after get the single unit, we can go further to detect the trend, or event
e.g. params are: dataset, delta, window size
'''
# choose one unit to dectec the trends
npF.Trend_Detection(df_seg[1], 279, 1000)
# choose the whole sequences 
npF.Trend_Detection(df_power, 279, 1000)

