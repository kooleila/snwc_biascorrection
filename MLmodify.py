import pandas as pd
import numpy as np
import time
import sys
import math

# modify time to sin/cos representation
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


# modify the pd frame used in ML training & realtime production
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
                'WS0bias','WG0bias','T1bias','WS1bias','WG1bias','RH1bias','RH_weight']
        elif param == 'WS':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','TA_PT1M_AVG',
                'WG_PT10M_MAX','fcdate','fmisid','obs_elevation','RHero','Tero','WGero','RH0bias',
                'T0bias','WG0bias','RH1bias','T1bias','WG1bias','WS1bias','WS_weight']
        elif param == 'WG':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
                'TA_PT1M_AVG','fcdate','fmisid','obs_elevation','RHero','WSero','Tero','RH0bias',
                'WS0bias','T0bias','RH1bias','WS1bias','T1bias','WG1bias']
	
	# dealing with missing values. 
	# dt this point all rows with Na values are removed 
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
        # reorder the data to be sure that the order is the same in training/prediction
        if param == 'T2m':
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
                'TA_PT1M_AVG', 'Tero', 'ElevD', 'T0bias', 'day_sin', 'day_cos',
                'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_T']]
        elif param == 'WS':
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
                'WS_PT10M_AVG', 'WSero', 'ElevD', 'WS0bias', 'day_sin', 'day_cos',
                'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_WS']]
        elif param == 'RH':
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
                'RH_PT1M_AVG', 'RHero', 'ElevD', 'RH0bias', 'day_sin', 'day_cos',
                'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_RH']]
        elif param == 'WG':
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'obs_lat', 'obs_lon',
                'WG_PT10M_MAX', 'WGero', 'ElevD', 'WG0bias', 'day_sin', 'day_cos',
                'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
	
	# modify data precision from f64 to f32
        data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)
        return data
