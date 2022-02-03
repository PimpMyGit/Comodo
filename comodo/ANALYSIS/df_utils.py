###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'df_utils.py': Pandas DataFrame utils functions

###############################################################################

import numpy as np
import pandas as pd
import scipy as spy
import seaborn as sns
import matplotlib.pyplot as plt

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST

from . import plot_utils as PLT

###############################################################################

"""
Generate DataFrame
"""

def makedf(entries, columns=None):
    entries = [[entry] for entry in entries] if type(entries[0]) is not list else entries
    columns = ['value_'+str(i) for i in range(len(entries[0]))] if columns==None or len(columns)!=len(entries[0]) else columns
    return pd.DataFrame(entries, columns=columns)

#-----------------------------------------------------------------------------#

""" Set indexes """

def set_datetime_index(df, column):
    return df.set_index(df[column].astype('datetime64[ns]'))

#-----------------------------------------------------------------------------#

""" Support columns """

def add_order_column(df, order_col_name='order'):
    df[order_col_name] = [i for i in range(df.shape[0])]
    return df

#-----------------------------------------------------------------------------#

"""
Read / Write / Append a .csv File
"""

def rcsv(filename, sep=',', header=[]):
    if header:
        return pd.read_csv(PATHS._CSV_DIR + (filename if UTILS.end_with(filename, '.csv') else filename+'.csv'), sep=sep, names=header)
    else:
        return pd.read_csv(PATHS._CSV_DIR + (filename if UTILS.end_with(filename, '.csv') else filename+'.csv'), sep=sep)

def wcsv(df, filename, sep=',', index=False):
    df.to_csv(PATHS._CSV_DIR + (filename if UTILS.end_with(filename, '.csv') else filename+'.csv'), sep=sep, index=index)

def acsv(df, filename, sep=',', index=False):
    df.to_csv(PATHS._CSV_DIR + (filename if UTILS.end_with(filename, '.csv') else filename+'.csv'), mode='a', sep=sep, index=index, header=False)

#-----------------------------------------------------------------------------#

"""
Has NAN
"""

def has_nan(df, mode='nan|None'):
    condition = lambda c: c is np.nan or c == np.nan or c is None or c == None
    if mode == 'nan':
        condition = lambda c: c is np.nan or c == np.nan
    elif mode == 'None':
        condition = lambda c: c is None or c == None
    nan_dict = DICT.filter_value({column: len(LIST.lfilter(df[column], condition)) for column in list(df.columns)}, lambda value: value > 0)
    nan_dict = DICT.merge(nan_dict, {'nan_percentage': sum(list(nan_dict.values()))/df.shape[0]}) if nan_dict else nan_dict
    return nan_dict
    

#-----------------------------------------------------------------------------#

"""
Apply
"""
# non funziona
def apply(df, input_column_s, lambda_fun, output_name_s=None, ignore_warning=True):
    out_calc = df[input_column_s].apply(func=lambda_fun, axis=(0 if type(input_column_s) is list else 1))
    if output_name_s==None:
        if type(input_column_s) is str:
            if type(out_calc[0]) in (list, tuple):
                if ignore_warning:
                    UTILS.throw_msg('warning', """Try to set object of type \''+str(type())+'\' as df attribute as str() value. 
                                                    Set \'output_name_s\' to assign them a single parameter or 
                                                    set \'ignore_warning\' to \'False\' to avoid this and stop with error.""")
                    df[input_column_s] = out_calc
                else:
                    UTILS.throw_msg('error', 'can\'t set object of type \''+str(type())+'\' as df attribute. Set \'ignore_warning\' to \'True\' to ignore this an keep the str() value')
                    return None
            elif type(out_calc[0]) is dict:
                df = add_cols(df, out_calc)
            return df
        else:
            UTILS.throw_msg('error', '\'output_name_s\' must be specified')
            return None
    else:
        return add_cols(df, out_calc, names=output_name_s)

#-----------------------------------------------------------------------------#

"""
Add columns
"""
def add_cols(df, cols, names=None, prefix=''):
    if type(cols[0]) is dict:
        keys = list(cols[0].keys())
        for key in keys:
            df[prefix+key] = [col[key] for col in cols]
    elif type(cols[0]) is list:
        if len(cols)==len(names):
            for col,name in zip(cols,names):
                df[prefix+name]=col
        else:
            UTILS.throw_msg('error', 'new values are \''+str(len(cols))+'\' but passed names are \''+str(len(names))+'\'')
            return None
    else:
        if type(names) is str:
            df[prefix+names] = cols
        else:
            UTILS.throw_msg('error', '\'names\' must be defined as one')
            return None
    return df

"""
Append rows
"""
def append(df1, df2):
    return df1.append(df2)

#-----------------------------------------------------------------------------#

"""
To list of dict
"""
def list_dict(df, cols=None, group_cols=None):
    cols = list(df.columns) if cols==None else cols
    list_dict = []
    if group_cols:
        group_cols = LIST.lvalues(group_cols)
        gdf = df.groupby(group_cols)
        for gk in list(gdf.groups.keys()):
            g = gdf.get_group(gk)
            d = {}
            for col in LIST.lfilter(list(g.columns), lambda c: c not in group_cols):
                d[col] = LIST.lset(list(g[col]))
            lgk = LIST.lvalues(gk)
            for ig,col in enumerate(group_cols):
                d[col] = lgk[ig]
            list_dict.append(d)                   
    else:
        list_dict = list(df.apply(lambda row: dict(row[cols]), axis=1))
    return list_dict

#-----------------------------------------------------------------------------#

"""
Find functions - big selection on every attributes
"""

def o_find(df, col, o_val):
    dfout = df.copy()
    if type(o_val) is str:
        o_val = [o_val]
    if o_val[0] == '!':
        dfout = dfout.loc[~dfout[col].isin(o_val[1:])].copy()
    elif o_val[0] == 're':
        dfout = dfout.loc[dfout[col].str.contains(o_val[1])].copy()
    else:
        dfout = dfout.loc[dfout[col].isin(o_val)].copy()
    return dfout

def n_find(df, col, n_val):
    dfout = df.copy()
    if type(n_val) is int or type(n_val) is float:
        n_val = [n_val]
    if type(n_val[0]) is str:
        op = n_val[0]
        n_val = n_val[1:]
        if op == '=':
            dfout = dfout.loc[dfout[col].isin(n_val)].copy()
        elif op == '>':
            dfout = dfout.loc[dfout[col] > n_val[0]].copy()
        elif op == '>=':
            dfout = dfout.loc[dfout[col] >= n_val[0]].copy()
        elif op == '<':
            dfout = dfout.loc[dfout[col] < n_val[0]].copy()
        elif op == '<=':
            dfout = dfout.loc[dfout[col] <= n_val[0]].copy()
        elif op == '!':
            dfout = dfout.loc[~dfout[col].isin(n_val)].copy()
        elif op == 'in':
            dfout = dfout.loc[(dfout[col] >= n_val[0]) & (dfout[col] <= n_val[1])].copy()
        elif op == '!in':
            dfout = dfout.loc[~((dfout[col] >= n_val[0]) & (dfout[col] <= n_val[1]))].copy()
        else:
            UTILS.throw_msg('error', 'operation \'' + op + '\' not valid')        
    else:
        dfout = dfout.loc[dfout[col].isin(n_val)].copy()
    return dfout

def find(df, cts={}):
    dft = dict(df.dtypes)
    dfout = df.copy()
    for ccol in cts:
        if ccol in df.columns:
            t = dft[ccol]
            cval = cts[ccol]
            if t == 'O': # tipo stringa
                dfout = o_find(dfout, ccol, cval).copy()
            elif t == 'int64' or t == 'float64':
                dfout = n_find(dfout, ccol, cval).copy()
        else:
            UTILS.throw_msg('error', 'column' + ccol + 'not in dataframe')
    return dfout

def find_lambda(df, lambda_fun):
    return df[df.apply(lambda_fun, axis=1)]

#-----------------------------------------------------------------------------#

"""
Base descriptive statistic
"""

def describe(df, x, print_res=True, print_plot=True, hue=None, colors=None, over=False, fsize=None, title=''):
    drs = spy.stats.describe(df[x])
    values = pd.Series([np.int64(v) for v in list(df[x])])
    drp = dict(values.describe())
    drt = { 'n obs':  drs[0],
            'mean':  drs[2],
            'variance':  drs[3],
            'std':  drp['std'] if 'std'in list(drp.keys()) else 'NaN',
            'min':  drp['min'] if 'min'in list(drp.keys()) else 'NaN',
            'q25':  drp['25%'] if '25%'in list(drp.keys()) else 'NaN',
            'q50':  drp['50%'] if '50%'in list(drp.keys()) else 'NaN',
            'q75':  drp['75%'] if '75%'in list(drp.keys()) else 'NaN',
            'max':  drp['max'] if 'max'in list(drp.keys()) else 'NaN',
            'range':  drp['max']-drp['min'] if 'max' in list(drp.keys()) else 'NaN',
            'skewness':  drs[4],
            'kurtosis':  drs[5] }
    if print_res:
        print('# Describe variable: ' + x + '\n')
        [print('• ' + r + ' :   ' + str(drt[r])) for r in drt]
        print('\n')
    if print_plot:
        PLT.describe(df, x, hue=hue, colors=colors, over=over, fsize=fsize, title=title)
    return drt

#-----------------------------------------------------------------------------#

###############################################################################