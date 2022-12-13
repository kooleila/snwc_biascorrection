import feather
import pandas as pd

df = pd.read_feather('eka2021.ftr', columns=None, use_threads=True);
print(df.shape)
print(df.isnull().sum())

drop_rows = df.dropna( how='all',
                          subset=['RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT10M_MAX'])
print(drop_rows.shape)
print(drop_rows.isnull().sum())

