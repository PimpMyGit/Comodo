###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'plot_utils.py': Seaborn and MatPlotLib utils functions

###############################################################################

import numpy as np
import pandas as pd
import scipy as spy
import seaborn as sns
import matplotlib.pyplot as plt

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT

###############################################################################

default_params = {
    'fsize' : MODULES._SNS_4x3_FIG_SIZE,
    'title' : '',
    'color' : '#2F528F',
    'hue': None,
    'palette': MODULES._SNS_COLOR_PALETTE
}

###############################################################################

"""
Describe - Distribution plot
"""

def describe(df, attribute, hue=None, colors=None, over=False, fsize=None, title=''):
    values = pd.Series([np.int64(v) for v in list(df[attribute])])
    if not over:
        plt.figure()
    if hue != None:
        df = df.sort_values(hue)
        hue = [str(v) for v in list(df[hue])]
    palette = MODULES._SNS_COLOR_PALETTE if not colors else colors
    fsize = fsize if fsize!=None else MODULES._SNS_7x7_FIG_SIZE
    sns.displot(x=values, kde=True, hue=hue, palette=palette, aspect=(fsize[0]/fsize[1]), element="step").set(title=title)

#-----------------------------------------------------------------------------#

"""
Heat Map - Distance Matrix
"""

def heatmap(matrix, fsize=None, title=''):
    MODULES.set_sns_fsize(MODULES._SNS_7x7_FIG_SIZE if fsize==None else fsize)
    ax = plt.axes()
    mat = sns.heatmap(np.array(matrix), ax=ax)
    ax.set_title(title)
    plt.show()
    MODULES.restore_sns_fsize()

#-----------------------------------------------------------------------------#

"""
Time Series
"""

def ts(df, x, y, params=default_params, ticks_rotation=25):
    p = DICT.merge(params, default_params)
    MODULES.set_sns_fsize(p['fsize'])
    sns.lineplot(data=df, x=x, y=y, hue=p['hue'], palette=p['palette'])
    plt.title(p['title'])
    plt.xticks(rotation=ticks_rotation)

#-----------------------------------------------------------------------------#

"""
Scatterplot
"""

def sp(df, x, y, params=default_params, ticks_rotation=25):
    p = DICT.merge(params, default_params)
    MODULES.set_sns_fsize(p['fsize'])
    sns.scatterplot(data=df, x=x, y=y, hue=p['hue'], palette=p['palette'])
    plt.title(p['title'])
    plt.xticks(rotation=ticks_rotation)

###############################################################################