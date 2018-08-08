"""
Python module for filter out the data with low variance
take in 
"""
from sklearn import preprocessing 
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def kill_all_x(x, df=None):
	"""
	This function is for remove 'all-zero', 'all-one', 'all-x' cases
	especially in 'DINT', 'BOOL' type date.
	"""
	if df.empty:
		return "the data frame is empty."
	num_attr = df.columns.size
	dropall = []
	for i in range(0,num_attr):
		index = df.columns[i]
		if(all(df[index]==x)):
			dropall.append(df.columns[i])
	df.drop(dropall, axis = 1, inplace= True)
	return df

def keep_x_var(x, df=None):
	"""
	this function is for filter out the x% low variance data
	the df type should be dataframe
	"""
	if df.empty:
		return "the data frame is empty."
	
	sel = VarianceThreshold(threshold=(x* (1 - x)))
	res = sel.fit_transform(df.values)
	return res
