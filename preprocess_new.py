import pandas as pd
import sqlite3
import datetime
from time import gmtime, strftime
import time 
import requests
import io
import feather
import re
import os

# read obs from Smartmet server
def readobs_all(starttime,endtime,t_reso,producer,obs_param):
    url = 'http://smartmet.fmi.fi/timeseries'
    params = "wmo,fmisid,latitude,longitude,time,elevation,"+obs_param
     
    payload1 = {
       "keyword": "snwc",
       "tz":"utc", # miksi tämä oli kommentoitu pois?!?
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
   
def merge_data(kk): #alku,loppu):
	start_time = time.time() # laske suoritusaika
	#aikasql = pd.to_datetime(alku).strftime("%Y%m")
		
	# Read sqlite query results into a pandas DataFrame
	con = sqlite3.connect("/data/hietal/MNWC_data/"+kk)     #MNWC"+aikasql+".sqlite")
	#df = pd.read_sql_query("SELECT DISTINCT SID FROM FC",con)
	#df = pd.read_sql_query("SELECT leadtime,parameter,SID,lat,lon,model_elevation,validdate,MNWC_preop_det FROM FC limit 2000000", con)

	df = pd.read_sql_query("SELECT leadtime, SID, lat, lon, model_elevation, validdate, fcdate, \
		avg(case when PARAMETER = 'D10m' THEN MNWC_preop_det END) AS D10M, \
		avg(case when PARAMETER = 'T2m' THEN MNWC_preop_det END) AS T2M, \
		avg(case when PARAMETER = 'S10m'  THEN MNWC_preop_det END) AS S10M, \
		avg(case when PARAMETER = 'RH2m'  THEN MNWC_preop_det END) AS RH2M, \
		avg(case when PARAMETER = 'Pmsl'  THEN MNWC_preop_det END) AS PMSL, \
		avg(case when PARAMETER = 'Q2m'  THEN MNWC_preop_det END) AS Q2M, \
		avg(case when PARAMETER = 'CClow'  THEN MNWC_preop_det END) AS CCLOW, \
		avg(case when PARAMETER = 'Gmax'  THEN MNWC_preop_det END) AS GMAX, \
		COUNT(*) AS rows_n FROM FC WHERE parameter IN ('T2m','D10m','S10m','RH2m','Pmsl','Q2m','CClow','Gmax') \
		GROUP BY leadtime, SID, LAT, LON, MODEL_ELEVATION,VALIDDATE, FCDATE", con)
	con.close()
	print(df['validdate'][1:5])
	obstime = pd.to_datetime(df['validdate'], unit='s') # aika havaintohakua varten
	df['validdate'] = pd.to_datetime(df['validdate'], origin='unix', unit='s',utc=True) 
	print(df['validdate'][1:5])
	#exit()
	#datetime.datetime.utcfromtimestamp(df['validdate']) #pd.to_datetime(df['validdate'], unit='s', utc=True).dt.tz_convert('Europe/Helsinki')
	#print(df['validdate'])
	df['fcdate'] = pd.to_datetime(df['fcdate'], origin='unix', unit='s', utc=True)

	# define times for obs retrieval
	alku = min(obstime) #df['validdate'].min()
	loppu = max(obstime) #df['validdate']. max()
	print(alku)
	print(loppu)
	 

	# runtime info
	print("%s seconds" % (time.time() - start_time))
	obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WSP_PT10M_AVG,WG_PT1H_MAX'  # WG_10M_MAX muutettu 1h!, WSP_PT10M_AVG
	AA = readobs_all(alku,loppu,'1h','observations_fmi',obs_param).content
	obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WS_PT10M_AVG,WG_PT1H_MAX'
	BB = readobs_all(alku,loppu,'1h','foreign',obs_param).content

	colnames=['SID','fmisid','obs_lat','obs_lon','validdate','obs_elevation','RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT1H_MAX']
	AAA = pd.read_csv(io.StringIO(AA.decode('utf-8')),names=colnames,header=None )
	BBB = pd.read_csv(io.StringIO(BB.decode('utf-8')),names=colnames,header=None )
	OBS = pd.concat([AAA, BBB])
	print(OBS.head(10))
	#print(OBS.dtypes)
	OBS['validdate'] = pd.to_datetime(OBS['validdate'], utc=True)
	print(OBS['validdate'].min())
	print(OBS['validdate'].max())
	#exit()
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
	#df_new['TD2M'] = df_new['TD2M'] - 273.15
	df_new['Tero'] = df_new['T2M'] - df_new['TA_PT1M_AVG']
	df_new['RHero'] = df_new['RH2M'] - df_new['RH_PT1M_AVG']
	df_new['WSero'] = df_new['S10M'] - df_new['WS_PT10M_AVG']
	df_new['WGero'] = df_new['GMAX'] - df_new['WG_PT1H_MAX']
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
	
	#tiputa tieto sqlite haettavien lkm 
	df_new = df_new.drop(['rows_n'],axis=1)
	# tiputa rivit, joilta kaikki havainnot puuttuu (esim ei toiminnassa olevat asemat)
	df_new = df_new.dropna( how='all', subset=['RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT1H_MAX'])
	#df_new.to_csv("/data/hietal/allmnwc.csv",index=False)

	return df_new

# Hae vuosidata

#lf <- list.files("/data/hietal/MNWC_data")
"""
listf = os.listdir("/data/hietal/MNWC_data")
listkk = list(filter(lambda x:'2020' in x, listf))
#print(listkk)

#print(len(listkk))
#data = merge_data(listkk[0])
data = merge_data(listkk[6]) #q12
###print(listkk[6])
#for looppi 6kk datojen vuoden sqlite taulujen yli
#for x in range(1,6): #len(listkk)):
#THIS IS NORMALLY USED; JUST MODIFIFED TO WORKK FOR 1 MOnth
for x in range(7,len(listkk)):
	print(listkk[x])
	tmp_data = merge_data(listkk[x])
	data = pd.concat([data,tmp_data])

# poista rivit, joissa KAIKKI havainnot puuttuvaa (esim asemat ei ole enää käytössä jne)
#data = data.dropna( how='all', subset=['RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT10M_MAX'])

data = data.reset_index()
data.to_feather('utcmnwc2020q34.ftr')

###############
# käy läpi muut vuodet
listf = os.listdir("/data/hietal/MNWC_data")
listkk = list(filter(lambda x:'2021' in x, listf))
data = merge_data(listkk[0])
for x in range(1,6):
        print(listkk[x])
        tmp_data = merge_data(listkk[x])
        data = pd.concat([data,tmp_data])

data = data.reset_index()
data.to_feather('utcmnwc2021q12.ftr')

data = merge_data(listkk[6])
for x in range(7,len(listkk)):
        print(listkk[x])
        tmp_data = merge_data(listkk[x])
        data = pd.concat([data,tmp_data])

data = data.reset_index()
data.to_feather('utcmnwc2021q34.ftr')

# 2022
listf = os.listdir("/data/hietal/MNWC_data")
listkk = list(filter(lambda x:'2022' in x, listf))
#data = merge_data(listkk[0])
#for x in range(1,6):
#        print(listkk[x])
#        tmp_data = merge_data(listkk[x])
#        data = pd.concat([data,tmp_data])
#
#data = data.reset_index()
#data.to_feather('utcmnwc2022q12.ftr')

data = merge_data(listkk[6])
for x in range(7,len(listkk)):
        print(listkk[x])
        tmp_data = merge_data(listkk[x])
        data = pd.concat([data,tmp_data])

data = data.reset_index()
data.to_feather('utcmnwc2022q34.ftr')
"""
# 2023
listf = os.listdir("/data/hietal/MNWC_data")
listkk = list(filter(lambda x:'2023' in x, listf))
print(listkk)
#exit()
data = merge_data(listkk[0])
#for x in range(1,6):
#        print(listkk[x])
#        tmp_data = merge_data(listkk[x])
#        data = pd.concat([data,tmp_data])

data = data.reset_index()
data.to_feather('utcmnwc202302.ftr')

#data = merge_data(listkk[6])
#for x in range(7,len(listkk)):
#        print(listkk[x])
#        tmp_data = merge_data(listkk[x])
#        data = pd.concat([data,tmp_data])

#data = data.reset_index()
#data.to_feather('utcmnwc2021q34.ftr')






#alku = '2021-01-01 00:00:00'
#loppu = '2021-01-31 23:00:00'
#data = merge_data(alku,loppu)

#print(data.head(20))

# modaa, niin että voi hakea vuoden kerrallaan

	
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
#print("%s seconds" % (time.time() - start_time))

"""
def main():
    options, remainder = getopt.getopt(sys.argv[1:],[],['starttime=','endtime=','outfile','help'])

    for opt, arg in options:
        if opt == '--help':
            print('preprocess_mnwc.py starttime=<path> outfile=<outfile>')
            exit()
        elif opt == '--starttime':
                starttime = arg
        elif opt == '--endtime':
                endtime = arg
        elif opt == '--outfile':
                outfile = arg

    # Exit with error message if not all command line arguments are specified
    try:
        starttime, endtime, outfile
    except NameError:
        print('ERROR! Not all input parameters specified: ')
        exit()

    #alku = '2021-01-01 00:00:00'
    #loppu = '2021-01-31 23:00:00'
    aikasql = pd.to_datetime(starttime).strftime("%Y%m")

if __name__ == "__main__":
"""

