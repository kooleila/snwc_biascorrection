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
from MLmodify import modify 

filen = '/home/users/hietal/statcal/python_projects/MNWC_data/MLdata/'

# import training datasets
ds1 = pd.read_feather(filen + 'mnwc2021q12.ftr', columns=None, use_threads=True);
ds2 = pd.read_feather(filen + 'mnwc2021q34.ftr', columns=None, use_threads=True);
ds3 = pd.read_feather(filen + 'mnwc2020q12.ftr', columns=None, use_threads=True);
ds4 = pd.read_feather(filen + 'mnwc2020q34.ftr', columns=None, use_threads=True);

df = pd.concat([ds1,ds2,ds3,ds4])

print(sorted(df['leadtime'].unique()))

# plot basics
#print(df.head(5))
#print(df['SID'].unique())
#print(len(df['SID'].unique()))
#print(df.shape)
#print(df.isnull().sum())

###param = 'T2m' # 'RH' 'WS' 'WG'
### names in mnwc data T2M RH2M S10M GMAX
param = 'RH'

#print(param)
dada = modify(df,param)
dada = dada[dada.leadtime != 0]

#print(dada.shape)
#print(dada.columns)
#print(sorted(dada['leadtime'].unique()))
#print(dada.isnull().sum())

# divide data to train/validation dataset
Xdf_train, Xdf_test, y_train, y_test = train_test_split(dada, dada.iloc[:,16], test_size=0.10, random_state=42)

# remove columns not needed for training
if param == 'T2m':
        remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','oldB_T']
elif param == 'WS':
        remove = ['SID','validdate','WS_PT10M_AVG', 'WSero','oldB_WS']
elif param == 'RH':
        remove = ['SID','validdate','RH_PT1M_AVG', 'RHero'] #,'oldB_RH']
elif param == 'WG':
        remove = ['SID','validdate','WG_PT10M_MAX', 'WGero']
X_train = Xdf_train.drop(remove, axis=1)
X_test = Xdf_test.drop(remove, axis=1)

# Test RandomForestRegressor 
# choose some 'default values' for hyperparameters
start_time = time.time()
print(start_time)
RFregressor = RandomForestRegressor(n_estimators=100, max_features='sqrt',random_state=42, max_depth=10, min_samples_split=2, bootstrap=True, min_samples_leaf=1)
# create an xgboost regression model
regressor = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
### lataa malli
###regressor = joblib.load("xgb_WSv1.joblib")

print(X_train.columns)
print(y_train)
RFregressor.fit(X_train,y_train)
regressor.fit(X_train, y_train)
print("%s seconds" % (time.time() - start_time))
# save the RF model
joblib.dump(RFregressor, "rf_RHv1.joblib")
joblib.dump(regressor, "xgb_RHv1.joblib")

#Calculate the temperature predictions of training and test sets 
y_trRFP = RFregressor.predict(X_train)
y_tsRFP = RFregressor.predict(X_test)
y_trP = regressor.predict(X_train)
y_tsP = regressor.predict(X_test)

print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))
print('RF train error:', mean_squared_error(y_train,y_trRFP,squared=False))
print('RF test error:', mean_squared_error(y_test,y_tsRFP,squared=False))

#print('Raw model test error:', mean_squared_error(Xdf_test['T2M'],Xdf_test['TA_PT1M_AVG'],squared=False))
#print('Raw model test error:', mean_squared_error(Xdf_test['S10M'],Xdf_test['WS_PT10M_AVG'],squared=False))
print('Raw model test error:', mean_squared_error(Xdf_test['GMAX'],Xdf_test['WS_PT10M_AVG'],squared=False))

#print(X_train.columns)
#print(sorted(Xdf_train['leadtime'].unique()))

