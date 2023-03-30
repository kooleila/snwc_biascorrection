import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import sys,os,getopt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from MLmodify import modify
import shap


def main():
	options, remainder = getopt.getopt(sys.argv[1:],[],['param=','help'])
    
	for opt, arg in options:
		if opt == '--help':
			print('MLshap.py param=T2m, WS, WG or RH')
			exit()
		elif opt == '--param':
			param = arg
	print(param)	
# define what parameter
#param = 'WG' # WS T2m RH WG
# load in the models 
	filen = '/home/users/hietal/statcal/python_projects/MNWC_data/MLdata/'
	model = joblib.load(filen + 'xgb_' + param + 'v1.joblib')
	
# read in the training data
# import training datasets
	ds1 = pd.read_feather(filen + 'eka2021.ftr', columns=None, use_threads=True);
	ds2 = pd.read_feather(filen + 'toka2021.ftr', columns=None, use_threads=True);
	ds3 = pd.read_feather(filen + 'kolmas2021.ftr', columns=None, use_threads=True);
	ds4 = pd.read_feather(filen + 'eka2020.ftr', columns=None, use_threads=True);
	ds5 = pd.read_feather(filen + 'toka2020.ftr', columns=None, use_threads=True);
	
	df = pd.concat([ds1,ds2,ds3,ds4,ds5])
	
	print(df.columns)
	dada = modify(df,param)
	
	# remove 0h 
	dada = dada[dada.leadtime != 0]
	
	print(dada.columns)
	
	# set the target parameter (F-O difference/model error) 
	y_test = dada.iloc[:,16]
	
	# remove columns not needed for training
	if param == 'T2m':
		remove = ['SID','validdate','TA_PT1M_AVG', 'Tero']
	elif param == 'WS':
		remove = ['SID','validdate','WS_PT10M_AVG', 'WSero']
	elif param == 'RH':
		remove = ['SID','validdate','RH_PT1M_AVG', 'RHero']
	elif param == 'WG':
		remove = ['SID','validdate','WG_PT10M_MAX', 'WGero']
	X_test = dada.drop(remove, axis=1)
	# produce shap pics 
	# take random set from test data
	X_sub = X_test.sample(frac=0.0005)
	print(X_sub.size)
	shap_values = shap.Explainer(model).shap_values(X_sub)
	shap.summary_plot(shap_values, X_sub, plot_type="bar", show = False)
	plt.savefig('bar_' + param + '.png')
	# another plot
	shap.summary_plot(shap_values, X_sub, show = False)
	plt.savefig('sum_' + param + '.png')
	
if __name__ == "__main__":
	main()


