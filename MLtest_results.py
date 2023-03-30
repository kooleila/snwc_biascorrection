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

def density_scatter( x , y, ax = None, sort = True, bins = 20, yla = '', xla = '', picname='scatterplot',  **kwargs)   :
	"""
	Scatter plot colored by 2d histogram
	"""
	if ax is None :
		fig , ax = plt.subplots()
	data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

	#To be sure to plot all data
	z[np.where(np.isnan(z))] = 0.0

	# Sort the points by density, so that the densest points are plotted last
	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, **kwargs )
	lineStart = x.min() 
	lineEnd = x.max() 
	#add diagonal
	ax.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
	plt.xlabel(xla)
	plt.ylabel(yla)

	###norm = Normalize(vmin = np.min(z), vmax = np.max(z))
	norm = Normalize(vmin = 0, vmax = 0.02)
	cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
	cbar.ax.set_ylabel('Density')
	plt.savefig(picname + '.png')
	return ax


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
	del ds1, ds2


	#df = pd.read_feather(filen + 'eka2022.ftr', columns=None, use_threads=True);
	#regressor1 = joblib.load(filen + 'xgb_' + param + 'v1.joblib')
	regressor2 = joblib.load(filen2 + 'xgb_' + param + '_tuned23.joblib')
	#random = joblib.load(filen + 'rf_WGv1.joblib")

	#ddff.iloc[:,27].values

	df_res = df #.copy()
	#print(df_res.columns)
	# create the "old" bias correction
	Tkerroin = [1,1, 0.9, 0.9, 0.8, 0.7,0,0,0]
	RHkerroin = [1,1, 0.9, 0.8, 0.7, 0.7,0,0,0]
	WSkerroin = [1,1, 0.8, 0.7, 0.6, 0.6,0,0,0]

	# check that you have all the leadtimes (0-9)
	ajat = sorted(df_res['leadtime'].unique().tolist())
	print(ajat)

	if param == 'T2m':
		df_res['oldB'] = df_res['T2M']
		kerroin = Tkerroin
		ok = 9 # defines the index of mnodel values
		wk = 33 # defines the index of 1h bias
	elif param == 'RH':
		df_res['oldB'] = df_res['RH2M']
		kerroin = RHkerroin
		ok = 11
		wk = 34
	elif param == 'WS':
		df_res['oldB'] = df_res['S10M']
		kerroin = WSkerroin
		ok = 10
		wk = 35

	df_res['weight'] = 0

	# set the correct weight for respective leadtime in data
	if param != 'WG':
		for i in range(0,6):
			df_res.weight[df_res['leadtime']==ajat[i]] = kerroin[i]
			print(i)
			print(kerroin[i])

		if param == 'T2m':
			df_res['oldB'] = df_res['T2M'] - df_res['weight']*df_res['T1bias']
		if param == 'RH':
			df_res['oldB'] = df_res['RH2M'] - df_res['weight']*df_res['RH1bias']
		if param == 'WS':
			df_res['oldB'] = df_res['S10M'] - df_res['weight']*df_res['WS1bias']
	if param == 'WG':
		df_res['oldB'] = 0
	#print(df.columns)
	#exit()

	if param == 'WS':
		m = 7
		df.drop(df[df['WS_PT10M_AVG'] >= 45].index,inplace=True)


	dada = modify(df,param)
	df_res = modify_res(df_res,param)
	# remove 0h 
	dada = dada[dada.leadtime != 0]
	dada = dada[dada.leadtime != 1]
	df_res = df_res[df_res.leadtime != 0]
	df_res = df_res[df_res.leadtime != 1]
	#print(len(dada))
	#print(len(df_res))

	print(dada.columns)
	print(df_res.columns)
	#exit()
	# set the target parameter (F-O difference/model error)
	y_test = dada.iloc[:,12]
	#print(y_test)


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
	print(X_test.columns)
	#exit()
	# make predictions for xgb (regressor) versions
	#y_tsP1 = regressor1.predict(X_test)
	y_tsP2 = regressor2.predict(X_test)
	#print(X_test.columns)
	#print(len(y_tsP2))
	#print(len(df_res))
	print(df_res.columns)
	# exit()
	
	#print('Random forest test error:', mean_squared_error(y_test,y_rfP,squared=False))
	#print('Gradient boosting validation error:', mean_squared_error(y_test,y_tsP1,squared=False))
	print('Tuned validation error:', mean_squared_error(y_test,y_tsP2,squared=False))
	if param == 'WS':
		print('Raw model test error:', mean_squared_error(df_res['S10M'],df_res['WS_PT10M_AVG'],squared=False))
	elif param == 'T2m':
		print('Raw model test error:', mean_squared_error(df_res['T2M'],df_res['TA_PT1M_AVG'],squared=False))
	elif param == 'RH':
		print('Raw model test error:', mean_squared_error(df_res['RH2M'],df_res['RH_PT1M_AVG'],squared=False))
	elif param == 'WG':
		print('Raw model test error:', mean_squared_error(df_res['GMAX'],df_res['WG_PT1H_MAX'],squared=False))
	#exit()
	df_res['bc'] = y_tsP2
	print(len(df_res))
	if param == 'T2m':
		m = 5 # defines the place of the mnwc parameter
		# plot scatterplot
		df_res.drop(df_res[df_res['fmisid']==117419].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117425].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117433].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117413].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117416].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117430].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117475].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117434].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117431].index, inplace=True)
		df_res.drop(df_res[df_res['fmisid']==117427].index, inplace=True)
	if param == 'WS':
		m = 7
		df_res.drop(df_res[df_res['WS_PT10M_AVG'] >= 45].index,inplace=True)
		###df_res.loc[df_res[''] == "male", "gender"] = 1
	if param == 'RH':
		m = 7
	if param == 'WG':
		m = 10
		df_res.drop(df_res[df_res['WG_PT1H_MAX'] >= 60].index,inplace=True)
	print(len(df_res))
	"""
	# divide to seasons 
	summer = df_res[df_res["month"].isin([6,7,8])]
	autumn = df_res[df_res["month"].isin([9,10,11])]
	spring = df_res[df_res["month"].isin([3,4,5])]
	winter = df_res[df_res["month"].isin([12,1,2])]
	
	
	ddff = winter
	yyy = ddff.iloc[:,m].values # model data
	xxx = ddff.iloc[:,14].values #-y_ts obs
	ddd = ddff.iloc[:,27].values
	#print(xxx)
	#print(yyy)
	density_scatter(xxx, yyy, bins = [20,20],yla= 'MNWC',xla='Observed',picname=param + 'DJFmnwc')
	yyy = yyy - ddd
	density_scatter(xxx, yyy, bins = [20,20],yla= 'XGboost',xla='Observed',picname=param + 'DJFxgb')
	ddff = spring
	yyy = ddff.iloc[:,m].values # mnwc data
	xxx = ddff.iloc[:,14].values # havainto
	ddd = ddff.iloc[:,27].values # xgb:n tuottama bc korjaus
	#print(xxx)
	#print(yyy)
	density_scatter(xxx, yyy, bins = [20,20],yla= 'MNWC',xla='Observed',picname=param + 'MAMmnwc')
	yyy = yyy - ddd
	density_scatter(xxx, yyy, bins = [20,20],yla= 'XGboost',xla='Observed',picname=param + 'MAMxgb')
	ddff = autumn
	yyy = ddff.iloc[:,m].values # model data
	xxx = ddff.iloc[:,14].values #-y_ts obs
	ddd = ddff.iloc[:,27].values
	#print(xxx)
	#print(yyy)
	density_scatter(xxx, yyy, bins = [20,20],yla= 'MNWC',xla='Observed',picname=param + 'SONmnwc')
	yyy = yyy - ddd
	density_scatter(xxx, yyy, bins = [20,20],yla= 'XGboost',xla='Observed',picname=param + 'SONxgb')
	ddff = summer
	yyy = ddff.iloc[:,m].values # model data
	xxx = ddff.iloc[:,14].values #-y_ts obs
	ddd = ddff.iloc[:,27].values
	#print(xxx)
	#print(yyy)
	density_scatter(xxx, yyy, bins = [20,20],yla= 'MNWC',xla='Observed',picname=param + 'JJAmnwc')
	yyy = yyy - ddd
	density_scatter(xxx, yyy, bins = [20,20],yla= 'XGboost',xla='Observed',picname=param + 'JJAxgb')
	"""
	#exit()
	
	ajat = sorted(dada['leadtime'].unique().tolist())
	print(ajat)
	raw = [None] * len(ajat)
	xg2 = [None] * len(ajat)
	old = [None] * len(ajat)
	for i in range(0,len(ajat)):
		print(ajat[i])
		tmp = df_res[df_res['leadtime']==ajat[i]] # all the parameter
		##X_tmp = X_test[X_test['leadtime']==ajat[i]] 
		##X_tmp2 = X_test2[X_test2['leadtime']==ajat[i]]
		y_tmp = tmp.iloc[:,15] #.values #RHero --> mallivirhe
		##print(y_tmp)
		##y_xgb1 = regressor1.predict(X_tmp)
		###y_xgb2 = regressor2.predict(X_tmp)
		y_xgb2 = tmp['bc']
		#rf[i] = mean_squared_error(y_tmp,y_rf,squared=False)
		#xg1[i] = mean_squared_error(y_tmp,y_xgb1,squared=False)
		xg2[i] = mean_squared_error(y_tmp,y_xgb2,squared=False)
	
		if param == 'WS':
			raw[i] = mean_squared_error(tmp['S10M'],(tmp['S10M'].values-y_tmp),squared=False)
			old[i] = mean_squared_error(tmp['oldB'],(tmp['S10M'].values-y_tmp),squared=False)
		elif param == 'T2m':
			raw[i] = mean_squared_error(tmp['T2M'],(tmp['T2M'].values-y_tmp),squared=False)
			old[i] = mean_squared_error(tmp['oldB'],(tmp['T2M'].values-y_tmp),squared=False)
		elif param == 'RH':
			raw[i] = mean_squared_error(tmp['RH2M'],(tmp['RH2M'].values-y_tmp),squared=False)
			old[i] = mean_squared_error(tmp['oldB'],(tmp['RH2M'].values-y_tmp),squared=False)
		elif param == 'WG':
			raw[i] = mean_squared_error(tmp['GMAX'],(tmp['GMAX'].values-y_tmp),squared=False)

	print(raw)
	print(xg2)
	print(old)

	plt.figure(2)
	#fig, ax = plt.subplots(1)
	plt.plot(ajat, raw, 'r', label="mnwc")  
	plt.plot(ajat, xg2, 'b', label="XGB2") 
	plt.plot(ajat, old, 'y', label="old BC")
	#plt.plot(t, c, 'g') # plotting t, c separately 
	plt.title("NWC RMSE")
	plt.xlabel("leadtime")
	plt.ylabel("RMSE")
	plt.legend(loc="lower right")
	plt.grid()
	if param == 'T2m':
		plt.ylim(0,2)
	elif param == 'WS':
		plt.ylim(0,2)
	elif param == 'WG':
		plt.ylim(0,3)
	elif param == 'RH':
		plt.ylim(0,12)
	plt.savefig('leadt_' + param + '_rmse.png')

#plt.show()
if __name__ == "__main__":
        main()
