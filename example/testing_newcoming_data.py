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

def main():
	# this script is used for testing and give feedback to training set
	# =========Step 1 read in data============
	# load in data, take the TDMS data type as example
	tdms_file = TdmsFile("/media/sherry/新加卷/ubuntu/WZL-2018/Feintool Daten/FW-1-1/new material/AKF_SS-FW2-H04521-H05000.tdms")
	# tdms_file.groups()
	df_all = tdms_file.object('Untitled').as_dataframe()
	df_force = df_all['Stempel_1 (Formula Result)']
	df_stroke = df_all['Position_Ma']
	# sample time series data 
	df_f = df_force[80800:99000].reset_index(drop=True)
	df_s = df_stroke[80800:99000].reset_index(drop=True)

	# the training data read in
	segmentations = read_from_file()
   
    # =========step 2: extract the hub ===========
	# Extract all punches of the dataset
	SEH = npF.SegHub()
	df_punches_t = SEH.extract_hub(df_f, df_s)
	df_punches_t = df_punches_t.reset_index(drop=True)

	# =========Step 3: segmentation into trends=========
	punch_seg = SEH.segment_and_plot(df_punches_t[0].dropna(), 'l2')
	sub_punch_seg = SEH.segment_and_plot(punch_seg[4].dropna(), 'rbf', 0, 4)
	punch_seg[4] = sub_punch_seg[0]
	punch_seg[5] = sub_punch_seg[1]
	punch_seg[6] = sub_punch_seg[2]
	punch_seg[7] = sub_punch_seg[3]

	
	# =========Step 4: classification=========
	for i in range(0,8):
	    print("Trend:"+str(i+1))
	    s = SEH.Uniformation(punch_seg[i])
	    clusters = pd.read_csv("cluster_"+str(i)+".csv")
	    data_train= SEH.Uniformation(segmentations[i])
	    row,col=data_train.shape
	    col= min(len(s),col)
	    print("Result:.........")
	    s = s[0:col]
	    test = pd.DataFrame([s,s])
	    data_train = data_train.iloc[:,0:col]
	    # generate new clusters and save into the file
	    # you cannot direct use xxx = yyyy for tables
	    new_dataset = data_train.copy()
	    new_dataset.loc[row] = s.values
	    z = hac.linkage(new_dataset, 'ward')
	    result = SEH.print_clusters(data_train,z,3, plot = False)
	    pd.DataFrame(result).to_csv("cluster_"+str(i)+".csv",index=False)

	    SEH.classifier(data_train,clusters,test ,3)

	#==========Step 5: save the newly added file==========
	save_newdata(punch_seg)

def save_newdata(punch_seg):
	i = 0
	for row in punch_seg:
		path = "segmentation_"+str(i)+".csv"
		i = i+1
		with open(path,'a') as fd:
			fd.write(row)
def read_from_file():
    # sss.tocsv("xxx.csv")
    segmentations=[[],[],[],[],[],[],[],[],[]]
    for i in range(0,8):
        segmentations[i] = pd.read_csv("segmentation_"+str(i)+".csv")
   
    return segmentations		

main()
