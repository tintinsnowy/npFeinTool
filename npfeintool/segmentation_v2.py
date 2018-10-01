from nptdms import TdmsFile
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

class SegHub(object):

    def segment_and_plot(self, df, model='l1', count=0, n_bkps=5):
        '''
        recieves a segment as input and runs ruptures changeppoint detection algorithm to segment a
        punch into subsequences. Eventually, it plots the segments,
        saves the plot and returns a dataframe containing all segments numbered begining with 1
        '''
        dfs = pd.DataFrame()
        lower_bound = 0
        upper_bound = len(df)

        # change point detection
        #model = "l1"  # "l2", "rbf"

        data = df[lower_bound:upper_bound].values
        algo = rpt.Dynp(model=model, min_size=15, jump=15).fit(data)
        my_bkps_t = algo.predict(n_bkps)

        j = 0
        lower_bound_temp = lower_bound
     
        while j < n_bkps :
            df_temp = pd.DataFrame({j: df[my_bkps_t[j]:my_bkps_t[j+1]]})
            df_temp = df_temp.reset_index(drop=True)
            dfs = pd.concat([dfs, df_temp], axis=1)
            #df[lower_bound_temp:my_bkps_t[j]+lower_bound].plot()
            #lower_bound_temp = my_bkps_t[j]+lower_bound
            j = j + 1
        #plt.savefig('plot_' + model + '_' + str(count) + '.png')
        #plt.clf()
        #print(dfs)
        return dfs

    def extract_hub(self, df_force, df_stroke, start=3000, threshold=0.51):
        '''
        returns data frame containing each punch segment derived by the df_stroke timeseries data
        '''
        dfs = pd.DataFrame()
        flag = False
        begin_temp = 0
        count = 0
        end = len(df_stroke)
        for i in range(start, end):
            if not(flag) and df_stroke[i] > threshold:
                flag = True
                begin_temp = i
            elif flag and df_stroke[i] > threshold:
                continue
            elif flag and df_stroke[i] < threshold:
                #print(begin_temp,i)
                seg_temp = df_force[begin_temp:i].reset_index(drop=True)
                #seg_temp = (begin_temp, i)
                #segments.append(seg_temp)
                df_temp= pd.DataFrame({count: seg_temp})
                #df_temp = pd.DataFrame(seg_temp)
                #df_temp = df_temp.reset_index(drop=True)
                dfs = pd.concat([dfs,df_temp], axis =1)
                flag = False
                count = count + 1
            else:
                continue
        return dfs


    def plot_all(df_to_plot):
        '''
        plots all columns in the dataframe
        '''
        ax = None
        for index in df_to_plot:
            ax = sns.tsplot(ax=ax, data=df_to_plot[index].values, err_style="unit_traces")
        plt.savefig('test.png')

    def separate_subsequence(df_segments, target_segment):
        dfs = pd.DataFrame()
        
        for j in range(0, len(df_segments)):
            df_temp = pd.DataFrame({j: df_segments[j]})
            df_temp = df_temp.reset_index(drop=True)   
            dfs = pd.concat([dfs, df_temp], axis=1)

    def DTWDistance(self, s1, s2):
        '''
        plots all columns in the dataframe
        '''
        DTW = {}

        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')
        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            for j in range(len(s2)):
                dist = (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

        return math.sqrt(DTW[len(s1)-1, len(s2)-1])



