import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
from MLmodify_results import modify_res
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def main():
	options, remainder = getopt.getopt(sys.argv[1:],[],['param=','help'])

	for opt, arg in options:
		if opt == '--help':
			print('MLshap.py param=T2m, WS, WG or RH')
			exit()
		elif opt == '--param':
			param = arg
	print(param)


	# read in the data and the models
	filen = '/home/users/hietal/statcal/python_projects/MNWC_data/snwc_biascorrection.git/'
	filen2 = '/data/hietal/'
	ds1 = pd.read_feather(filen + 'utcmnwc2022q12.ftr', columns=None, use_threads=True);
	ds2 = pd.read_feather(filen + 'utcmnwc2022q34.ftr', columns=None, use_threads=True);
	df = pd.concat([ds1,ds2])
	
	#print(ds1.count)
	#print(ds1.describe())
	#print(ds1.info(verbose=True))
	#print(ds1.isnull().sum(axis = 0))

	#exit()
		
	regressor2 = joblib.load(filen2 + 'xgb_' + param + '_tuned23.joblib')

	dada = modify(df,param)
	df_res = modify_res(df,param)
	# remove 0h 
	dada = dada[dada.leadtime != 0]
	dada = dada[dada.leadtime != 1]
	df_res = df_res[df_res.leadtime != 0]
	df_res = df_res[df_res.leadtime != 1]
	#print(len(dada))
	#print(len(df_res))

	print(dada.columns)
	print('df_res:', df_res.columns)

	# set the target parameter (F-O difference/model error)
	y_test = dada.iloc[:,12]
	print(y_test)


	# remove columns not needed for training
	if param == 'T2m':
		remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','T0bias']
		#df = df[df['TA_PT1M_AVG'].notna()]
	elif param == 'WS':
		remove = ['SID','validdate','WS_PT10M_AVG', 'WSero','WS0bias']
		#df = df[df['WS_PT10M_AVG'].notna()]
	elif param == 'RH':
		remove = ['SID','validdate','RH_PT1M_AVG', 'RHero','RH0bias']
		#df = df[df['RH_PT1M_AVG'].notna()]
	elif param == 'WG':
		remove = ['SID','validdate','WG_PT1H_MAX', 'WGero','WG0bias']
		#df = df[df['WG_PT10M_MAX'].notna()]

	X_test = dada.drop(remove, axis=1)
	#print(X_test.columns)

	# make predictions for xgb (regressor) versions
	y_tsP2 = regressor2.predict(X_test)

	print('Tuned validation error:', mean_squared_error(y_test,y_tsP2,squared=False))
	if param == 'WS':
		print('Raw model test error:', mean_squared_error(df_res['S10M'],df_res['WS_PT10M_AVG'],squared=False))
	elif param == 'T2m':
		print('Raw model test error:', mean_squared_error(df_res['T2M'],df_res['TA_PT1M_AVG'],squared=False))
	elif param == 'RH':
		print('Raw model test error:', mean_squared_error(df_res['RH2M'],df_res['RH_PT1M_AVG'],squared=False))
	elif param == 'WG':
		print('Raw model test error:', mean_squared_error(df_res['GMAX'],df_res['WG_PT1H_MAX'],squared=False))


	ajat = sorted(dada['leadtime'].unique().tolist())
	print(ajat)
	df_res['xgb'] = y_tsP2
	# poista turha
	# car_df.drop(car_df.columns[[2, 5]], axis = 1, inplace = True)
	#T2m: 
	if param == 'T2m':
		df_res.drop(df_res.columns[[1,6,7,8,9,10,17,21,22,23,24]], axis=1,inplace=True)
		df_res = df_res.reset_index()
		df_res.to_csv('/data/hietal/T2m_res.csv', index=False)
	elif param == 'WS':
		df_res.drop(df_res.columns[[1,5,6,8,9,10,17,21,22,23,24]], axis=1,inplace=True)
		df_res = df_res.reset_index()
		df_res.to_csv('/data/hietal/WS_res.csv', index=False)
	elif param == 'WG':
		df_res.drop(df_res.columns[[1,5,6,7,8,9,17,21,22,23,24]], axis=1,inplace=True)
		df_res = df_res.reset_index()
		df_res.to_csv('/data/hietal/WG_res.csv', index=False)
	elif param == 'RH':
		df_res.drop(df_res.columns[[1,5,6,8,9,10,17,21,22,23,24]], axis=1,inplace=True)
		df_res = df_res.reset_index()
		df_res.to_csv('/data/hietal/RH_res.csv', index=False)
	print(df_res.columns)
	#exit()
	#print(df_res.head(10))
	#df_res = df_res.reset_index()
	#df_res.to_csv('/data/hietal/RH_res.csv', index=False)
	print(df_res.columns)
	#df_res.to_feather('T2m_res.ftr')		

#plt.show()
if __name__ == "__main__":
        main()
