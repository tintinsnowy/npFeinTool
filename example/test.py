from nptdms import TdmsFile
import numpy as np
import npfeintool as npF
from npfeintool import filterVar
import matplotlib.pyplot as plt


def main():
    """
    Simple example function which uses the npFeinTool package
    """
    # load in data, take the TDMS data type as example
    tdms_file = TdmsFile(".\\feintool_data\\FW-1-1\\new material\\AKF_SS-FW2-H04521-H05000.tdms")
    # preprocessing the data type 'DINT'
    df_force = tdms_file.object('Untitled').as_dataframe()

    # Step 1: kick out the "all zero", "all one" value
    # this Step is essential for selection meaningful selection

    df_temp = npF.kill_all_x(0, df_force)
    df_temp = npF.keep_x_var(0.8, df_temp)
    # findout the selected columns
    ix = np.isin(df_force.values[0,:], df_temp[0])
    column_indices = np.where(ix==True)
    df_force = df_force.iloc[:, column_indices[0]]
    '''
    Step 2: divide the periodical time series data into unit
    e.g. we select only one channel data to segementation
    Params are: dataset, windsize(the approx unit size)
    '''
    df_punch = df_force['Stempel_1 (Formula Result)']
    # df_seg = npF.SW_seg(df_punch, 500)


    len_s = 45000# len(s_fr)
    delta = 400#276
    win_size = 1000
    i = 38000
    j = i
    # s_fr.drop([len_s, len(s_fr) - 1]).to_csv('test.csv')
    df_punch.truncate(before=i, after=len_s - 1).to_csv('test.csv')
    # df['A'].truncate(before=2, after=4)

    while i < len_s:
        k = npF.Trend_Change(df_punch[j:i+win_size], delta)
        if k != -1:
            df_punch[j:j+k].plot()
            plt.savefig('plot' + str(i) + '.png')
            plt.clf()
            df_punch[j+k+1:i+win_size].plot()
            plt.savefig('plot' + str(i) + 'x.png')
            plt.clf()
            print(str(j) + " + " + str(j + k) + "; " + str(j+k+1) + " to " + str(i + win_size))
            j = i + win_size
        i += win_size

    plt.savefig('plot.png')

    # npF.Trend_Detection(df_punch[38000:45000], 400, 1000)
    # choose one unit to dectec the trends
    # npF.Trend_Detection(df_seg[2], 279, 1000)
    # choose the whole sequences
    # npF.Trend_Detection(df_power, 279, 1000)

main()
