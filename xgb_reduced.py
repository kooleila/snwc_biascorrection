import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import sys,os,getopt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import xgboost as xgb
from xgboost import XGBRegressor
from MLmodify import modify 

def main():
	options, remainder = getopt.getopt(sys.argv[1:],[],['param=','help'])

	for opt, arg in options:
		if opt == '--help':
			print('MLshap.py param=T2m, WS, WG or RH')
			exit()
		elif opt == '--param':
			param = arg
	print(param)

	#filen = '/home/users/hietal/statcal/python_projects/MNWC_data/MLdata/'
	filen = '/home/users/hietal/statcal/python_projects/MNWC_data/snwc_biascorrection.git/'

	# import training datasets
	ds1 = pd.read_feather(filen + 'utcmnwc2020q12.ftr', columns=None, use_threads=True);
	ds2 = pd.read_feather(filen + 'utcmnwc2020q34.ftr', columns=None, use_threads=True);
	ds3 = pd.read_feather(filen + 'utcmnwc2021q12.ftr', columns=None, use_threads=True);
	ds4 = pd.read_feather(filen + 'utcmnwc2021q34.ftr', columns=None, use_threads=True);

	df = pd.concat([ds1,ds2,ds3,ds4])

	print(sorted(df['leadtime'].unique()))

	# plot basics
	print(df.head(5))
	print(df.columns)
	print(df['SID'].unique())
	print(len(df['SID'].unique()))
	print(df.shape)
	#print(df.isnull().sum())
	dada = modify(df,param)
	dada = dada[dada.leadtime != 0]
	dada = dada[dada.leadtime != 1]
	print(dada.columns)
	print(len(dada))
	if param == 'WS':
		dada.drop(dada[dada['WS_PT10M_AVG'] >= 45].index,inplace=True)
	if param == 'WG':
		dada.drop(dada[dada['WG_PT1H_MAX'] >= 60].index,inplace=True)
	print(len(dada))
	# divide data to train/validation dataset
	Xdf_train, Xdf_test, y_train, y_test = train_test_split(dada, dada.iloc[:,12], test_size=0.10, random_state=42)
	
	# remove columns not needed for training
	if param == 'T2m':
		remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','T0bias']
		colby =	0.48
		learnr = 0.21
		maxD = 10
		n_est =	108
		reg_a =	0.71
		subsam = 0.9
	elif param == 'WS':
		remove = ['SID','validdate','WS_PT10M_AVG', 'WSero','WS0bias']
		colby = 0.58 # 0.73 # 0.62
		learnr = 0.1 # 0.44 # 0.053
		maxD = 12 # 12 # 15
		n_est = 127 # 108 # 147
		reg_a = 0.75 # 0.86 # 0.73
		subsam = 0.95 # 0.94 # 0.8
	elif param == 'RH':
		remove = ['SID','validdate','RH_PT1M_AVG', 'RHero','RH0bias']
		colby =	0.99
		learnr = 0.23
		maxD = 12
		n_est =	124
		reg_a =	0.06
		subsam = 0.5
	elif param == 'WG':
		remove = ['SID','validdate','WG_PT1H_MAX', 'WGero','WG0bias']
		colby =	 0.75 # 0.55
		learnr = 0.27 # 0.16
		maxD = 7 # 10
		n_est = 114 # 106
		reg_a =	0.04 # 0.27
		subsam = 0.61 # 0.8
	X_train = Xdf_train.drop(remove, axis=1)
	X_test = Xdf_test.drop(remove, axis=1)
	print(X_train.head(5))
	print(X_train.columns)
	#X_test.to_csv('/data/hietal/XGB/RT_data/X_test.csv', index = False)
	#exit()
	start_time = time.time()
	print(start_time)
	# create an xgboost regression model
	#regressor = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
	regressor = xgb.XGBRegressor(n_estimators=n_est, max_depth=maxD, eta=learnr, subsample=subsam, colsample_bytree=colby,alpha=reg_a)
### lataa malli
###regressor = joblib.load("xgb_WSv1.joblib")

	print(X_train.columns)
	print(y_train)
	regressor.fit(X_train, y_train)
	print("%s seconds" % (time.time() - start_time))
	# save the model
	joblib.dump(regressor,'/data/hietal/xgb_' + param + '_tuned23.joblib')	

	#Calculate the predictions of training and test sets 
	y_trP = regressor.predict(X_train)
	y_tsP = regressor.predict(X_test)

	print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
	print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))

	#print('Raw model test error:', mean_squared_error(Xdf_test['T2M'],Xdf_test['TA_PT1M_AVG'],squared=False))
	#print('Raw model test error:', mean_squared_error(Xdf_test['S10M'],Xdf_test['WS_PT10M_AVG'],squared=False))
	print('Raw model test error:', mean_squared_error(Xdf_test['GMAX'],Xdf_test['WG_PT1H_MAX'],squared=False))
	#print('Raw model test error:', mean_squared_error(Xdf_test['RH2M'],Xdf_test['RH_PT1M_AVG'],squared=False))

if __name__ == "__main__":
	main()
