from nptdms import TdmsFile
import npfeintool as npF
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.cluster.hierarchy as hac
import math
import numpy as np
from random import sample
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.cluster.hierarchy import fcluster

def save_to_file():
    """
    This example is designed for testing 
    """
    # ============Step 1: Read in data============
    # load in data, take the TDMS data type as example
    tdms_file = TdmsFile("/media/sherry/新加卷/ubuntu/WZL-2018/Feintool Daten/FW-1-1/new material/AKF_SS-FW2-H04521-H05000.tdms")
    # tdms_file.groups()
    df_all = tdms_file.object('Untitled').as_dataframe()
    df_force = df_all['Stempel_1 (Formula Result)']
    df_stroke = df_all['Position_Ma']

    # ============Step 2: Extract each punch shape============
    SEH = npF.SegHub()
    # Init lists and dfs
    segmentation_1 = []
    segmentation_2 = []
    segmentation_3 = []
    segmentation_4 = []
    segmentation_5 = []
    sub_punch_1 =[]
    sub_punch_2 =[]
    sub_punch_3 =[]
    sub_punch_4 =[]
    # Extract all punches of the dataset
    df_punches = SEH.extract_hub(df_force, df_stroke, end = 100000)
    #print(df_punches.describe())
    df_punches = df_punches.reset_index(drop=True)

    # ============Step 3: separate into subsegmentation============
    x=0
    for i in df_punches:
        # first changepoint detection on whole punch
        punch_seg = SEH.segment_and_plot(df_punches[i].dropna(), 'l2')
        # second go further to get divide the fifth sequence
        segmentation_1.append(np.asarray(punch_seg[0]))
        segmentation_2.append(np.asarray(punch_seg[1]))
        segmentation_3.append(np.asarray(punch_seg[2]))
        segmentation_4.append(np.asarray(punch_seg[3]))
        segmentation_5.append(np.asarray(punch_seg[4]))
        sub_punch_seg = SEH.segment_and_plot(punch_seg[4].dropna(), 'rbf', 7 + i, 4)
        # append to corresponding list
        sub_punch_1.append(np.asarray(sub_punch_seg[0]))
        sub_punch_2.append(np.asarray(sub_punch_seg[1]))
        sub_punch_3.append(np.asarray(sub_punch_seg[2]))
        sub_punch_4.append(np.asarray(sub_punch_seg[3]))
        #sub_segmentation.append(sub_punch_seg)
        x = 1+x

    # Save into files in case 
    pd.DataFrame(segmentation_1).to_csv("segmentation_0.csv",index=False)
    pd.DataFrame(segmentation_2).to_csv("segmentation_1.csv",index=False)
    pd.DataFrame(segmentation_3).to_csv("segmentation_2.csv",index=False)
    pd.DataFrame(segmentation_4).to_csv("segmentation_3.csv",index=False)
    pd.DataFrame(segmentation_5).to_csv("segmentation_4(1).csv",index=False)
    pd.DataFrame(sub_punch_1).to_csv("segmentation_4.csv",index=False)
    pd.DataFrame(sub_punch_2).to_csv("segmentation_5.csv",index=False)
    pd.DataFrame(sub_punch_2).to_csv("segmentation_6.csv",index=False)
    pd.DataFrame(sub_punch_2).to_csv("segmentation_7.csv",index=False)
    

def read_from_file():
    # sss.tocsv("xxx.csv")
    segmentations=[[],[],[],[],[],[],[],[],[]]
    for i in range(0,8):
        segmentations[i] = pd.read_csv("segmentation_"+str(i)+".csv")
   
    return segmentations

def save_models(segmentations):
    SEH = npF.SegHub()
    data_seg = [[],[],[],[],[],[],[],[],[],[]]
    for i in range (0,8):
        data_seg[i]= SEH.Uniformation(segmentations[i])
        z = hac.linkage(data_seg[i], 'ward')
        result = SEH.print_clusters(data_seg[i],z,3,False)
        pd.DataFrame(result).to_csv("cluster_"+str(i)+".csv",index=False)
    
def main():
    # save the processed sequences into file in case for some failure 
    save_to_file()
    segmentations = read_from_file()
    # Uniformation + cluster + save in the file
    save_models(segmentations)

main()
