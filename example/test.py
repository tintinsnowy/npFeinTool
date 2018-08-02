from  nptdmsnptdms  import
import  TdmsFileTdmsFil 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seglearn as sgl
from matplotlib.mlab import PCA
import npFeintool as npF

# load in data, take the TDMS data type as example
tdms_file = TdmsFile(".\\FW-1-1\\AKF-FW1-1-H1021-1500.tdms")
#preprocessing the data type 'DINT' 
df = tdms_file.object('DINT').as_dataframe()

'''
Step 1: kick out the "all zero", "all one" value
this Step is essential for selection meaningful selection

'''
df =  npF.kill_all_x(df, 0)
df = npF.keep_x_var(df, 0.8)

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

npF.Trend_Detection(df_seg[0], 279, 1000)