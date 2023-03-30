import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
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
from skopt.space import Real, Integer, Categorical
from skopt import dump, load
import sys,os,getopt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import xgboost as xgb
from xgboost import XGBRegressor
from MLmodify import modify 
from skopt import BayesSearchCV 
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

#def rmse_cv(model, X = X_train, y = y_train):
#	return np.sqrt(-cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = kf)).mean()

"""
def on_step(optim_result):

	#Callback meant to view scores after
	#each iteration while performing Bayesian
	#Optimization in Skopt
	score = opt.best_score_
	print("best score: %s" % score)
	if score >= -0.11:
		print('Interrupting!')
		return True

"""
def main():
	options, remainder = getopt.getopt(sys.argv[1:],[],['param=','help'])

	for opt, arg in options:
		if opt == '--help':
			print('MLshap.py param=T2m, WS, WG or RH')
			exit()
		elif opt == '--param':
			param = arg
	print(param)

	filen = '/home/users/hietal/statcal/python_projects/MNWC_data/MLdata/'

	# import training datasets
	ds1 = pd.read_feather(filen + 'mnwc2020q12.ftr', columns=None, use_threads=True);
	ds2 = pd.read_feather(filen + 'mnwc2020q34.ftr', columns=None, use_threads=True);
	ds3 = pd.read_feather(filen + 'mnwc2021q12.ftr', columns=None, use_threads=True);
	ds4 = pd.read_feather(filen + 'mnwc2021q34.ftr', columns=None, use_threads=True);

	df = pd.concat([ds1,ds2,ds3,ds4])
	
	print(sorted(df['leadtime'].unique()))

# plot basics
#print(df.head(5))
#print(df['SID'].unique())
	print(len(df['SID'].unique()))
	print(df.shape)
	print(df.isnull().sum())

	dada = modify(df,param)
	dada = dada[dada.leadtime != 0]
	dada = dada[dada.leadtime != 1]
	print(len(dada))
	# divide data to train/validation dataset
	if param == 'WS':
		dada.drop(dada[dada['WS_PT10M_AVG'] >= 45].index,inplace=True)
	if param == 'WG':
		dada.drop(dada[dada['WG_PT1H_MAX'] >= 60].index,inplace=True)

	Xdf_train, Xdf_test, y_train, y_test = train_test_split(dada, dada.iloc[:,12], test_size=0.10, random_state=42)
	#print(y_test)
	print(len(dada))
	# remove columns not needed for training
	if param == 'T2m':
		remove = ['SID','validdate','TA_PT1M_AVG', 'Tero', 'T0bias']
	elif param == 'WS':
		remove = ['SID','validdate','WS_PT10M_AVG', 'WSero', 'WS0bias']
	elif param == 'RH':
		remove = ['SID','validdate','RH_PT1M_AVG', 'RHero' , 'RH0bias']
	elif param == 'WG':
		remove = ['SID','validdate','WG_PT1H_MAX', 'WGero','WG0bias']
	X_train = Xdf_train.drop(remove, axis=1)
	X_test = Xdf_test.drop(remove, axis=1)

	start_time = time.time()
	print(start_time)
	print(X_test.columns)

	optimizer_kwargs = {'acq_func_kwargs':{"xi": 10, "kappa": 10}}
	space  = {'max_depth':Integer(5, 15),
		'learning_rate':Real(0.05, 0.55, "uniform"),
		'colsample_bytree':Real(0.1,1,'uniform'),
		'subsample': Real(0.4, 1, "uniform"),
		'reg_alpha': Real(1e-9, 1,'uniform'),
		'n_estimators': Integer(40, 160)}
	bsearch = BayesSearchCV(estimator = xgb.XGBRegressor(random_state=10), #GradientBoostingRegressor(random_state=10), 
	search_spaces = space, scoring='neg_mean_absolute_error',n_jobs=6, n_iter=10, cv=5, optimizer_kwargs=optimizer_kwargs)
	bsearch.fit(X_test,y_test)
	#parameter_over_iterations(bsearch)

	dump(bsearch,'/data/hietal/results_' + param + '_new.pkl')
	print("Best Score is: ", bsearch.best_score_, "\n")
	print("Best Parameters: ", bsearch.best_params_, "\n")

	#filen + 'mnwc2020q12.ftr', columns=None
# n_iter was 64

"""

	# create an xgboost regression model
	regressor = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
### lataa malli
###regressor = joblib.load("xgb_WSv1.joblib")

	print(X_train.columns)

	xgbr = xgb.XGBRegressor(objective = "reg:squarederror", n_jobs = -1, random_state = 0)

	# %%time
	# start = datetime.now()
	# print(start)
	kf = KFold(n_splits = 4, shuffle = True, random_state = 0)


	opt = BayesSearchCV(xgbr, 
			{	
			"learning_rate": Real(0.01, 1.0, 'uniform'),
			"n_estimators": Integer(50, 100),
			"max_depth": Integer(3, 13),
			"colsample_bytree": Real(0.1, 1,'uniform'),
			"subsample": Real(0.1, 1,'uniform'),
			"reg_alpha": Real(1e-9, 1,'uniform'),
			"reg_lambda": Real(1e-9, 1,'uniform'),
			"gamma": Real(0, 0.5)
			},
		n_iter = 10,  
		cv = kf,
		n_jobs = -1,
		scoring = "neg_root_mean_squared_error",
		random_state = 0
 		)

	res = opt.fit(X_test, y_test)
	dump(res,'results_wg.pkl')	

# end = datetime.now()
#	print(end)
	print("Best Score is: ", opt.best_score_, "\n")
	print("Best Parameters: ", opt.best_params_, "\n")
	
# xgbr2 = opt.best_estimator_
# xgbr2
######
	# bayes tuning
	params={'min_child_weight': (0, 50,),
	'max_depth': (0, 10),
	'subsample': (0.5, 1.0),
	'colsample_bytree': (0.5, 1.0),
	'reg_lambda':(1e-5,100,'log-uniform'),
	'reg_alpha':(1e-5,100,'log-uniform'),
	'learning-rate':(0.01,0.2,'log-uniform')
	}
	bayes = BayesSearchCV(xgb.XGBRegressor(),params,n_iter=2,scoring='neg_mean_squared_error',cv=2,random_state=42)
	res=bayes.fit(X_test,y_test)
	print(res.best_params_)



	params={'min_child_weight': (0, 50,),
		'max_depth': (0, 50),
		'subsample': (0.5, 1.0),
		'colsample_bytree': (0.5, 1.0),
		'reg_lambda':(1e-5,100,'log-uniform'),
		'reg_alpha':(1e-5,100,'log-uniform'),
		'learning-rate':(0.01,0.3,'log-uniform')
	}

	bayes = BayesSearchCV(xgb.XGBRegressor(),params,n_iter=10,scoring='neg_mean_squared_error',cv=5,random_state=42)
	res=bayes.fit(X_test,y_test)
	print(res.best_params_)
	dump(res,'results_t2m.pkl')

	#regressor.fit(X_train, y_train)
	#print("%s seconds" % (time.time() - start_time))
	# save the model
	#joblib.dump(regressor, 'xgb_' + param + 'v2.joblib')

	#Calculate the predictions of training and test sets 
	#y_trP = regressor.predict(X_train)
	#y_tsP = regressor.predict(X_test)

	#print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
	#print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))

	#print('Raw model test error:', mean_squared_error(Xdf_test['T2M'],Xdf_test['TA_PT1M_AVG'],squared=False))
	#print('Raw model test error:', mean_squared_error(Xdf_test['S10M'],Xdf_test['WS_PT10M_AVG'],squared=False))
	#print('Raw model test error:', mean_squared_error(Xdf_test['GMAX'],Xdf_test['WS_PT10M_AVG'],squared=False))
"""

if __name__ == "__main__":
	main()
