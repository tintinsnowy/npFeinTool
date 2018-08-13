'''
Functions for Stream Data Segmentation from TDMS Files.
'''

from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from LocalExtrema import detect_peaks



def bool_seg(tdms_file):
    '''
    Segmentation using BOOL Channel.
    Returns the lower and upper bound indices of each punch.
    '''
    df_seg = []         #initialization
    count = 0

    channel = tdms_file.object('BOOL', '130U1 (EL1202).Channel 1.Input')       #get BOOL Trigger values
    data = channel.data

    peaks = detect_peaks(data, edge='falling')              # sliding window method
    len_peaks = len(peaks)
    width = round(min(np.diff(peaks)))

    while count < len_peaks:               #Get lower and upper bounds
        low = peaks[count]
        high = low+width
        #seg = data[int(low): int(high)]
        df_seg.append([low, high])
        count = count + 1
    return df_seg



def data_seg(data, indices):
    '''
    Segments the input data into individual parts dependent on the input indices.
    Returns an array with each row being one punch
    '''
    seg_data = []
    for index in range(0, len(indices)):
        seg_data.append(data[int(indices[index][0]):int(indices[index][1])])
    return seg_data



def hub_characterization(hub_array):
    '''
    Adds attributes to each punch data.
    '''
    characteristics = []
    hubs_characteristics = []
    len_seg = len(hub_array)

    for index in range(0, len_seg):
        characteristics = [np.mean(hub_array[index]), np.var(hub_array[index]), np.max(hub_array[index]), np.min(hub_array[index])]
        hubs_characteristics.append(characteristics)
        characteristics = []

    return hubs_characteristics

