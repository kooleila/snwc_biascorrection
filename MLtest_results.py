import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
from MLmodify import modify

# define what parameter
param = 'WG' # WS T2m RH WG
# load in the models 
regressor = joblib.load("xgb_WGv1.joblib")
random = joblib.load("rf_WGv1.joblib")

# read in the data
filen = '/home/users/hietal/statcal/python_projects/MNWC_data/MLdata/'
df = pd.read_feather(filen + 'eka2022.ftr', columns=None, use_threads=True);

# create the "old" bias correction
Tkerroin = [0.9, 0.9, 0.8, 0.7,0,0,0,0,0]
RHkerroin = [0.9, 0.8, 0.7, 0.7,0,0,0,0,0]
WSkerroin = [0.8, 0.7, 0.6, 0.6,0,0,0,0,0]

# check that you have all the leadtimes (0-9)
ajat = sorted(df['leadtime'].unique().tolist())

if param == 'T2m':
	df['oldB'] = df['T2M']
	kerroin = Tkerroin
elif param == 'RH':
	df['oldB'] = df['RH2M']
	kerroin = RHkerroin
elif param == 'WS':
	df['oldB'] = df['S10M']
	kerroin = WSkerroin
	
df['weight'] = 0

# set the correct weight for respective leadtime in data
if param != 'WG':
	for i in range(0,4):
		df.weight[df['leadtime']==ajat[i]] = kerroin[i]
		#df.weight[df['leadtime']==ajat[i]] = kerroin[i]
		#df.weight[df['leadtime']==ajat[i]] = kerroin[i]

	#df['oldB'] = df['T2M'] - df['weight']*df['T0bias']
	#df['oldB'] = df['RH2M'] - df['weight']*df['RH0bias']
	df['oldB'] = df['S10M'] - df['weight']*df['WS0bias']

dada = modify(df,param)

# remove 0h 
dada = dada[dada.leadtime != 0]

print(dada.columns)

# set the target parameter (F-O difference/model error) 
y_test = dada.iloc[:,16]

# remove columns not needed for training
if param == 'T2m':
	remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','oldB']
elif param == 'WS':
	remove = ['SID','validdate','WS_PT10M_AVG', 'WSero','oldB']
elif param == 'RH':
	remove = ['SID','validdate','RH_PT1M_AVG', 'RHero','oldB']
elif param == 'WG':
	remove = ['SID','validdate','WG_PT10M_MAX', 'WGero']
X_test = dada.drop(remove, axis=1)

# make predictions for RF model (random) and xgb (regressor)
y_tsP = regressor.predict(X_test)
y_rfP = random.predict(X_test)
print(X_test.columns)
#print(y_test)

print('Random forest test error:', mean_squared_error(y_test,y_rfP,squared=False))
print('Gradient boosting test error:', mean_squared_error(y_test,y_tsP,squared=False))
if param == 'WS':
	print('Raw model test error:', mean_squared_error(dada['S10M'],dada['WS_PT10M_AVG'],squared=False))
elif param == 'T2m':
	print('Raw model test error:', mean_squared_error(dada['T2M'],dada['TA_PT1M_AVG'],squared=False))
elif param == 'RH':
	print('Raw model test error:', mean_squared_error(dada['RH2M'],dada['RH_PT1M_AVG'],squared=False))
elif param == 'WG':
	print('Raw model test error:', mean_squared_error(dada['GMAX'],dada['WG_PT10M_MAX'],squared=False))

ajat = sorted(dada['leadtime'].unique().tolist())
print(len(ajat))
raw = [None] * len(ajat)
rf = [None] * len(ajat)
xg = [None] * len(ajat)
old = [None] * len(ajat)
for i in range(0,len(ajat)):
	print(ajat[i])
	tmp = dada[dada['leadtime']==ajat[i]] # all the parameter
	X_tmp = X_test[X_test['leadtime']==ajat[i]] 
	y_tmp = tmp.iloc[:,16] #.values
	y_rf = random.predict(X_tmp)
	y_xgb = regressor.predict(X_tmp)
	rf[i] = mean_squared_error(y_tmp,y_rf,squared=False)
	xg[i] = mean_squared_error(y_tmp,y_xgb,squared=False)
	if param == 'WS':
		raw[i] = mean_squared_error(tmp['S10M'],(tmp['S10M'].values-y_tmp),squared=False)
		old[i] = mean_squared_error(tmp['oldB_WS'],(tmp['S10M'].values-y_tmp),squared=False)
	elif param == 'T2m':
		raw[i] = mean_squared_error(tmp['T2M'],(tmp['T2M'].values-y_tmp),squared=False)
		old[i] = mean_squared_error(tmp['oldB_T'],(tmp['T2M'].values-y_tmp),squared=False)
	elif param == 'RH':
		raw[i] = mean_squared_error(tmp['RH2M'],(tmp['RH2M'].values-y_tmp),squared=False)
		old[i] = mean_squared_error(tmp['oldB_RH'],(tmp['RH2M'].values-y_tmp),squared=False)
	elif param == 'WG':
		raw[i] = mean_squared_error(tmp['GMAX'],(tmp['GMAX'].values-y_tmp),squared=False)

#print(raw)
#print(rf)
#print(old)

plt.plot(ajat, raw, 'r', label="mnwc")  
plt.plot(ajat, rf, 'b', label="RF")  
plt.plot(ajat, xg, 'g', label="XGB") 
plt.plot(ajat, old, 'y', label="old BC")
#plt.plot(t, c, 'g') # plotting t, c separately 
plt.title("NWC RMSE")
plt.xlabel("leadtime")
plt.ylabel("RMSE")
plt.legend(loc="upper left")
#plt.plot(x, y)
plt.grid()
plt.savefig('MLnwcxgb.png')

plt.show()
