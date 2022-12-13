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
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
import xgboost as xgb
from xgboost import XGBRegressor


# import training datasets
ds1 = pd.read_feather('eka2021.ftr', columns=None, use_threads=True);
ds2 = pd.read_feather('toka2021.ftr', columns=None, use_threads=True);
ds3 = pd.read_feather('kolmas2021.ftr', columns=None, use_threads=True);
ds4 = pd.read_feather('eka2020.ftr', columns=None, use_threads=True);
ds5 = pd.read_feather('toka2020.ftr', columns=None, use_threads=True);

df = pd.concat([ds1,ds2,ds3,ds4,ds5])

#print(sorted(df['leadtime'].unique()))

# plot basics
#print(df.head(5))
#print(df['SID'].unique())
#print(len(df['SID'].unique()))
#print(df.shape)
#print(df.isnull().sum())

# modify time to sin/cos
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

# modify data
def modify(data,param):
	#data['validdate'] = pd.to_datetime(data['validdate'])
	#data = data.assign(diff_elev=data.model_elevation-data.elevation)
	if param == 'T2m':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
		'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','RHero','WSero','WGero','RH0bias',
		'WS0bias','WG0bias','RH1bias','WS1bias','WG1bias','T1bias']
                #'oldB_RH','oldB_WS','T_weight','WS_weight','RH_weight']
	elif param == 'RH':
		remove = ['index','model_elevation','lat','lon','TA_PT1M_AVG','WS_PT10M_AVG',
                'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','Tero','WSero','WGero','T0bias',
		'WS0bias','WG0bias','T1bias','WS1bias','WG1bias','RH1bias']
	elif param == 'WS':
		remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','TA_PT1M_AVG',
                'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','RHero','Tero','WGero','RH0bias',
		'T0bias','WG0bias','RH1bias','T1bias','WG1bias','WS1bias']
	elif param == 'WG':
		remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
                'TA_PT1M_AVG','fcdate','fmisid','obs_elevation','RHero','WSero','Tero','RH0bias',
		'WS0bias','T0bias','RH1bias','WS1bias','T1bias','WG1bias']
	
	data = data.drop(remove, axis=1)
    
	# modify time to sin/cos representation
	data = data.assign(month=data.validdate.dt.month)
	data = data.assign(day=data.validdate.dt.day)
	data = data.assign(hour=data.validdate.dt.hour)
	data = encode(data, 'day', 365)
	data = encode(data, 'month', 12)
	data = encode(data, 'hour', 24)
	data = data.drop(['month','day','hour'], axis=1)
	# Normalize values between 0-1
	#datarm = dataset.copy()
	#print(data.columns)
	# reorder data 
	data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
       'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
       'TA_PT1M_AVG', 'Tero', 'ElevD', 'T0bias', 'day_sin', 'day_cos',
       'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]

	##min_max_scaler = preprocessing.MinMaxScaler()
	##data.iloc[:,6:20] = min_max_scaler.fit_transform(data.iloc[:,6:20])
	data = data.dropna()
	return data  

param = 'T2m' # 'RH' 'WS' 'WG'
dada = modify(df,param)
dada = dada[dada.leadtime != 0]

#print(dada.shape)
#print(dada.columns)
#print(sorted(dada['leadtime'].unique()))
#print(dada.isnull().sum())

# divide data to train/validation dataset
Xdf_train, Xdf_test, y_train, y_test = train_test_split(dada, dada.iloc[:,16], test_size=0.10, random_state=42)
# remove columns not needed for training
remove = ['SID','validdate','TA_PT1M_AVG', 'Tero']
X_train = Xdf_train.drop(remove, axis=1)
X_test = Xdf_test.drop(remove, axis=1)

# Test RandomForestRegressor 
# choose some 'default values' for hyperparameters
###regressor = RandomForestRegressor(n_estimators=100, max_features='sqrt',random_state=42, max_depth=10, min_samples_split=2, bootstrap=True, min_samples_leaf=1)
# create an xgboost regression model
regressor = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)

start_time = time.time()
#print(start_time)
print(X_train.columns)
print(y_train)
regressor.fit(X_train, y_train)
print("%s seconds" % (time.time() - start_time))
# save the RF model
joblib.dump(regressor, "xgb_v1.joblib")

#Calculate the temperature predictions of training and test sets 
y_trP = regressor.predict(X_train)
y_tsP = regressor.predict(X_test)
print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))
print('Raw model test error:', mean_squared_error(Xdf_test['T2M'],Xdf_test['TA_PT1M_AVG'],squared=False))

#print(X_train.columns)
#print(sorted(Xdf_train['leadtime'].unique()))

