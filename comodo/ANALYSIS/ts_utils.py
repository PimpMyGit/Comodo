###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'ts_utils.py': Time-Series analysis utils functions

###############################################################################

import math
import numpy as np
import pandas as pd
import scipy as spy
import seaborn as sns
import matplotlib.pyplot as plt

import multiprocessing as mp

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST

from . import df_utils as DF
from . import plot_utils as PLT

###############################################################################

"""
Test stazionarieta ADF
"""

def adf_stationarity(df, ts):
    adf_test = adfuller(df[ts].values)
    print('ADF Statistic:', adf_test[0])
    print('p-value:', adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print('\t', key, ':', value)
    return adf_test


"""
Differenzazione semplice o stagionale
"""

def diff(l, delta, mode='simple', dropna=True):
    l = pd.Series(l) if type(l) is not pd.Series else l
    if mode == 'simple':
        dl = diff(l.diff(), delta-1, 'simple', dropna) if delta>1 else l.diff()
    elif mode == 'seasonal':
        dl = l.diff(delta)
    else:
        UTILS.throw_msg('error', 'mode param must be in [\'simple\', \'seasonal\']')
        return None
    return dl.dropna() if dropna else dl

#-----------------------------------------------------------------------------#

"""
Model comparison
"""

def accuracy(model, trdf, tsdf):
    pass

def mae(model, trdf, tsdf, ts):
    truth = tsdf[ts].values
    prediction = predict(model, trdf, tsdf).values
    return np.sum(abs(truth - prediction)) / tsdf.shape[0]

def mape(model, trdf, tsdf, ts):
    truth = tsdf[ts].values
    prediction = predict(model, trdf, tsdf).values
    return np.sum([abs((t - p) / t) for t,p in zip(truth, prediction)]) * (100 / tsdf.shape[0])

def mse(model, trdf, tsdf, ts):
    truth = tsdf[ts].values
    prediction = predict(model, trdf, tsdf).values
    return np.sum([math.pow((t - p), 2) for t,p in zip(truth, prediction)]) / tsdf.shape[0]  

def rmse(model, trdf, tsdf, ts):
    return math.sqrt(mse(model, trdf, tsdf, ts))

#-----------------------------------------------------------------------------#

"""
SARIMAX Models
"""

def fitting(model, train_df):
    return model.predict(start=0, end=train_df.shape[0])

def validating(model, train_df, test_df):
    return model.predict(strat=train_df.shape[0], end=test_df.shape[0])

def train_sarimax(df, ts, option, verbose=False):
    model = sm.tsa.statespace.SARIMAX(df[ts],
                                    order=option[0], seasonal_order=option[1],
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
    model = model.fit(disp=0)
    if verbose:
        UTILS.throw_msg('done', 'Trained model: ' + str(option))
    return model

def sarima_options(p_params,d_params,q_params,P_params,D_params,Q_params,s_params):
	options = []
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for P in P_params:
					for D in D_params:
						for Q in Q_params:
							for s in s_params:
								options.append( ((p,d,q), (P,D,Q,s)) )
	return options

def bruteforce_sarimax(df, ts, options, evaluation=('evaluation_lambda_fun', 'trdf', 'tsdf', 'df_csv_filename')):
    for i,option in enumerate(options):
        model = train_sarimax(df,ts,option, verbose=True)
        if evaluation:
            evaluation_df = pd.DataFrame([DICT.merge(compare_models(model, evaluation[1], evaluation[2], evaluation[0])[0], {'option': option})])
            _ = DF.wcsv(evaluation_df, evaluation[3]) if i == 0 else DF.acsv(evaluation_df, evaluation[3])
    return DF.rcsv(evaluation[3])

def predict(model, trdf, tsdf, exog=None):
    return model.get_prediction(start=trdf.shape[0], end=trdf.shape[0]+tsdf.shape[0]-1,exog=exog).predicted_mean

def compare_models(models, train_df, test_df=None, lambda_funs=[], plot_best=False):
    results = [UTILS.lambda_seq(model, train_df, test_df, lseq=lambda_funs) for model in LIST.lvalues(models)]
    return results

# ----------------------------------------------------------------------------#

"""
UCM Models
"""

def ucm_options(level, trend, cycle, seasonal, freq_seasonal, stochastic_level, stochastic_trend, stochastic_seasonal, stochastic_cycle):
    options = []
    for l in level:
        for t in trend:
            for c in cycle:
                for stc_l in stochastic_level:
                    for stc_t in stochastic_trend:
                        for stc_s in stochastic_seasonal:
                            for stc_c in stochastic_cycle:
                                for s, fs in zip(seasonal, freq_seasonal):
                                    for f in fs:
                                        options.append({
                                            'level': l,
                                            'trend': t,
                                            'cycle': c,
                                            'seasonal': s,
                                            'freq_seasonal': f,
                                            'stochastic_level': stc_l,
                                            'stochastic_trend': stc_t,
                                            'stochastic_seasonal': stc_s,
                                            'stochastic_cycle': stc_c
                                        })
    return options

def train_ucm(df, ts, verbose=False, **defargs):
    model = UnobservedComponents(df[ts], **defargs)    
    model = model.fit()
    if verbose: UTILS.throw_msg('done', 'Trained model: ' + str(defargs))    
    return model

def bruteforce_ucm(df, ts, options, evaluation=('evaluation_lambda_fun', 'trdf', 'tsdf', 'df_csv_filename')):
    for i,option in enumerate(options):
        model = train_ucm(df, ts, verbose=True, **option)
        if evaluation:
            evaluation_df = pd.DataFrame([DICT.merge(compare_models(model, evaluation[1], evaluation[2], evaluation[0])[0], {'option': option})])
            _ = DF.wcsv(evaluation_df, evaluation[3]) if i == 0 else DF.acsv(evaluation_df, evaluation[3])
    return DF.rcsv(evaluation[3])

###############################################################################

