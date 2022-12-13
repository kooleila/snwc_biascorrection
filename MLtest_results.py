import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.inspection import permutation_importance
from sklearn import metrics
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#from sklearn import preprocessing
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
#import xgboost as xgb
#from xgboost import XGBRegressor

df = pd.read_feather('eka2022.ftr', columns=None, use_threads=True);

Tkerroin = [0.9, 0.9, 0.8, 0.7,0,0,0,0,0]
RHkerroin = [0.9, 0.8, 0.7, 0.7,0,0,0,0,0]
WSkerroin = [0.8, 0.7, 0.6, 0.6,0,0,0,0,0]

ajat = sorted(df['leadtime'].unique().tolist())

df['oldB_T'] = df['T2M']
#df['oldB_RH'] = df['mnwc.RH2m']
#df['oldB_WS'] = df['mnwc.S10m']
df['T_weight'] = 0
#df['RH_weight'] = 0
#df['WS_weight'] = 0

for i in range(0,4):
    df.T_weight[df['leadtime']==ajat[i]] = Tkerroin[i]
    #df.RH_weight[df['leadtime']==ajat[i]] = RHkerroin[i]
    #df.WS_weight[df['leadtime']==ajat[i]] = WSkerroin[i]

df['oldB_T'] = df['T2M'] - df['T_weight']*df['T0bias']
#df['oldB_RH'] = df['mnwc.RH2m'] - df['RH_weight']*df['RH0bias']
#df['oldB_WS'] = df['mnwc.S10m'] - df['WS_weight']*df['WS0bias']


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
                'WS0bias','WG0bias','RH1bias','WS1bias','WG1bias','T1bias','T_weight']
                #'oldB_RH','oldB_WS','T_weight','WS_weight','RH_weight']
        elif param == 'RH':
                remove = ['index','model_elevation','lat','lon','TA_PT1M_AVG','WS_PT10M_AVG',
                'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','Tero','WSero','WGero','T0bias',
                'WS0bias','WG0bias','T1bias','WS1bias','WG1bias','RH1bias','T_weight']
        elif param == 'WS':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','TA_PT1M_AVG',
                'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','RHero','Tero','WGero','RH0bias',
                'T0bias','WG0bias','RH1bias','T1bias','WG1bias','WS1bias','T_weight']
        elif param == 'WG':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
                'TA_PT1M_AVG','fcdate','fmisid','obs_elevation','RHero','WSero','Tero','RH0bias',
                'WS0bias','T0bias','RH1bias','WS1bias','T1bias','WG1bias','T_weight']
        
        data = data.drop(remove, axis=1)
    
        # modify time to sin/cos representation
        data = data.assign(month=data.validdate.dt.month)
        data = data.assign(day=data.validdate.dt.day)
        data = data.assign(hour=data.validdate.dt.hour)
        data = encode(data, 'day', 365)
        data = encode(data, 'month', 12)
        data = encode(data, 'hour', 24)
        data = data.drop(['month','day','hour'], axis=1)
        
        data = data.dropna()
	# reorder data
        data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
       'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
       'TA_PT1M_AVG', 'Tero', 'ElevD', 'T0bias', 'day_sin', 'day_cos',
       'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_T']]
        return data  

#lataa malli
regressor = joblib.load("xgb_v1.joblib")
random = joblib.load("rf_v1.joblib")

param = 'T2m' # 'RH' 'WS' 'WG'
dada = modify(df,param)

#poista 0h ja jaa data
dada = dada[dada.leadtime != 0]
y_test = dada.iloc[:,16]
# remove columns not needed for training
remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','oldB_T']
X_test = dada.drop(remove, axis=1)
y_tsP = regressor.predict(X_test)
y_rfP = random.predict(X_test)
#print(X_test.columns)
#print(y_test)

y_tsP = regressor.predict(X_test)
y_rfP = random.predict(X_test)
print('Random forest test error:', mean_squared_error(y_test,y_rfP,squared=False))
print('Gradient boosting test error:', mean_squared_error(y_test,y_tsP,squared=False))
print('Raw model test error:', mean_squared_error(dada['T2M'],dada['TA_PT1M_AVG'],squared=False))

ajat = sorted(dada['leadtime'].unique().tolist())
print(len(ajat))
raw = [None] * len(ajat)
rf = [None] * len(ajat)
xg = [None] * len(ajat)
old = [None] * len(ajat)
for i in range(0,len(ajat)):
    print(ajat[i])
    tmp = dada[dada['leadtime']==ajat[i]] #kaikki parametrit
    X_tmp = X_test[X_test['leadtime']==ajat[i]] #vain ml koulutukseen
    y_tmp = tmp.iloc[:,16]#.values
    y_rf = random.predict(X_tmp)
    y_xgb = regressor.predict(X_tmp)
    rf[i] = mean_squared_error(y_tmp,y_rf,squared=False)
    xg[i] = mean_squared_error(y_tmp,y_xgb,squared=False)
    raw[i] = mean_squared_error(tmp['T2M'],(tmp['T2M'].values-y_tmp),squared=False)
    old[i] = mean_squared_error(tmp['oldB_T'],(tmp['T2M'].values-y_tmp),squared=False)

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
