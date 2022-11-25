import pandas as pd
import sqlite3
import datetime
from time import gmtime, strftime
import time 
import requests
import io

start_time = time.time() # laske suoritusaika

alku = '2021-01-01 00:00:00'
loppu = '2021-01-31 23:00:00'
aikasql = pd.to_datetime(alku).strftime("%Y%m")

def readobs_all(starttime,endtime,t_reso,producer,obs_param):
    url = 'http://smartmet.fmi.fi/timeseries'
    params = "wmo,fmisid,latitude,longitude,time,elevation,"+obs_param
     
    payload1 = {
       "keyword": "snwc",
       #"tz":"utc",
       "separator":",",
       "producer": producer,
       "precision": "double",
       "timeformat": "sql",
       "param": params,
       "starttime": starttime,   #"2020-09-18T00:00:00",
       "endtime": endtime,   #"2020-09-18T00:00:00",
       "timestep": t_reso,
       "format": "ascii"
    }
    r = requests.get(url, params=payload1)
    return r
    
obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WSP_PT10M_AVG,WG_PT10M_MAX'  #WSP_PT10M_AVG
AA = readobs_all(alku,loppu,'1h','observations_fmi',obs_param).content
obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WS_PT10M_AVG,WG_PT10M_MAX'
BB = readobs_all(alku,loppu,'1h','foreign',obs_param).content

colnames=['SID','fmisid','obs_lat','obs_lon','validdate','obs_elevation','RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT10M_MAX'] 
AAA = pd.read_csv(io.StringIO(AA.decode('utf-8')),names=colnames,header=None )
BBB = pd.read_csv(io.StringIO(BB.decode('utf-8')),names=colnames,header=None )
OBS = pd.concat([AAA, BBB])
#print(OBS.head(20))
#print(OBS.dtypes)
OBS['validdate'] = pd.to_datetime(OBS['validdate'])
#OBS['validdate'] = datetime.datetime.utcfromtimestamp(OBS['validdate']) #pd.to_datetime(OBS['validdate'], utc=True).dt.tz_convert('Europe/Helsinki')

print("%s seconds" % (time.time() - start_time))

# testi
# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("/data/hietal/MNWC_data/MNWC"+aikasql+".sqlite")
#df = pd.read_sql_query("SELECT DISTINCT SID FROM FC",con)
#df = pd.read_sql_query("SELECT leadtime,parameter,SID,lat,lon,model_elevation,validdate,MNWC_preop_det FROM FC limit 2000000", con)

df = pd.read_sql_query("SELECT leadtime, SID, lat, lon, model_elevation, validdate, fcdate, \
	avg(case when PARAMETER = 'D10m' THEN MNWC_preop_det END) AS D10M, \
	avg(case when PARAMETER = 'T2m' THEN MNWC_preop_det END) AS T2M, \
	avg(case when PARAMETER = 'CCtot'  THEN MNWC_preop_det END) AS CCTOT, \
	avg(case when PARAMETER = 'S10m'  THEN MNWC_preop_det END) AS S10M, \
	avg(case when PARAMETER = 'RH2m'  THEN MNWC_preop_det END) AS RH2M, \
	avg(case when PARAMETER = 'Pmsl'  THEN MNWC_preop_det END) AS PMSL, \
	avg(case when PARAMETER = 'Q2m'  THEN MNWC_preop_det END) AS Q2M, \
	avg(case when PARAMETER = 'CClow'  THEN MNWC_preop_det END) AS CCLOW, \
	avg(case when PARAMETER = 'Td2m'  THEN MNWC_preop_det END) AS TD2M, \
	avg(case when PARAMETER = 'Gmax'  THEN MNWC_preop_det END) AS GMAX, \
	COUNT(*) AS rows_n FROM FC WHERE parameter IN ('CCtot','T2m','D10m','S10m','RH2m','Pmsl','Q2m','CClow','Td2m','Gmax') \
	GROUP BY leadtime, SID, LAT, LON, MODEL_ELEVATION,VALIDDATE, FCDATE", con)
con.close()
#print(df['validdate'])
#print(df.dtypes)
df['validdate'] = pd.to_datetime(df['validdate'], unit='s') #datetime.datetime.utcfromtimestamp(df['validdate']) #pd.to_datetime(df['validdate'], unit='s', utc=True).dt.tz_convert('Europe/Helsinki')
#print(df['validdate'])
df['fcdate'] = pd.to_datetime(df['fcdate'], unit='s') #datetime.datetime.utcfromtimestamp(df['fcdate'])  #pd.to_datetime(df['fcdate'], unit='s', utc=True).dt.tz_convert('Europe/Helsinki')

#OBS['validdate'] = datetime.datetime.utcfromtimestamp(OBS['validdate']) #pd.to_datetime(OBS['validdate'], utc=True).dt.tz_convert('Europe/Helsinki')
#print(OBS['validdate'])

# time info
print("%s seconds" % (time.time() - start_time))

#print(OBS.head(20))
#print(df.head(20))

OBS['SID'].astype('int64') 

# merge model + obs
#OBS.to_csv("/data/hietal/testiobs.csv",index=False)
#df.to_csv("/data/hietal/testimnwc.csv",index=False)

df_new = pd.merge(df, OBS,  how='left', left_on=['SID','validdate'], right_on = ['SID','validdate'])

# calculate F-O errors
df_new['T2M'] = df_new['T2M'] - 273.15
df_new['TD2M'] = df_new['TD2M'] - 273.15
df_new['Tero'] = df_new['T2M'] - df_new['TA_PT1M_AVG']
df_new['RHero'] = df_new['RH2M'] - df_new['RH_PT1M_AVG']
df_new['WSero'] = df_new['S10M'] - df_new['WS_PT10M_AVG']
df_new['WGero'] = df_new['GMAX'] - df_new['WG_PT10M_MAX']
# elevation difference
df_new['ElevD'] = df_new['model_elevation'] - df_new['obs_elevation']

####Erottele analyysiajan 0h ja 1h virheet ja yhdistä
ajat = sorted(df_new['leadtime'].unique().tolist())
lt0 = df_new[df_new['leadtime']==0]
lt0 = lt0[['fcdate','SID','Tero','RHero','WSero','WGero']]
lt0.columns = ['fcdate', 'SID','T0bias','RH0bias','WS0bias','WG0bias']
lt1 = df_new[df_new['leadtime']==1]
lt1 = lt1[['fcdate','SID','Tero','RHero','WSero','WGero']]
lt1.columns = ['fcdate', 'SID','T1bias','RH1bias','WS1bias','WG1bias']
df_new[(df_new.leadtime != 0)] # & (df_new.ladtime != 1)]

df_new = pd.merge(df_new, lt0,  how='left', left_on=['SID','fcdate'], right_on = ['SID','fcdate'])
df_new = pd.merge(df_new, lt1,  how='left', left_on=['SID','fcdate'], right_on = ['SID','fcdate'])

df_new = df_new.drop(['rows_n'],axis=1)

df_new.to_csv("/data/hietal/allmnwc.csv",index=False)

print(df_new.columns)
#df_wide=pd.pivot(df, index=['leadtime','validdate','SID','lat','lon','model_elevation'], columns = 'parameter',values = 'MNWC_preop_det').reset_index() #Reshape from long to wide
# cobnvert table from narrow to wide
###df_w = (
###    df.pivot_table(
###        index=['leadtime','validdate','SID','lat','lon','model_elevation'],
###        columns='parameter',
###        values='MNWC_preop_det'
###    )
###    .reset_index()              # collapses multi-index
###    .rename_axis(None, axis=1)  # renames index
###)

#df_wide = pd.pivot_table(df,index=['leadtime','validdate','SID','lat','lon','model_elevation'],columns='parameter',values='MNWC_preop_det')

#print(df['validdate'][1].replace(second=0,microsecond=0))
#print(datetime.now())
#print(datetime.now().replace(second=0,microsecond=0))

#the_time = df['validdate'].replace(second=0, microsecond=0)
#aika = pd.Timestamp(df['validdate'].iloc)
#print(aika)
#df['DT'] = pd.to_datetime(df['DT'], unit='s', utc=True).dt.tz_convert('Europe/Amsterdam')
#df['hour_datetime'] = df['new'].dt.round('H')
#aika.floor(freq='H')
#print(aika)
#print(df_new.head(200))
#print(df['SID'].unique())
#print(len(df['SID'].unique()))
#print(df_shape.shape)
print("%s seconds" % (time.time() - start_time))

