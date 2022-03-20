###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'list_utils.py': List utils functions

###############################################################################

import math
import numpy as np
import pandas as pd

from . import general_utils as UTILS
from . import dict_utils as DICT

from ..ANALYSIS import df_utils as DF

###############################################################################

"""
Convert to numpy array if it is not yet
"""

def nparr(l):
    return np.array(l) if type(l) is not np.ndarray else l

#-----------------------------------------------------------------------------#

"""
Sort
"""

def lsort(l, reverse=False):
    return UTILS.lambda_vsafe(l, lambda x: x.sort(reverse=reverse))

#-----------------------------------------------------------------------------#

"""
Return as list of values
"""

def lvalues(val):
    return val if type(val) is list else list(val) if type(val) is tuple else [val]

def lvalue(l):
    return l[0] if len(l)==1 else l

#-----------------------------------------------------------------------------#

"""
Base descriptive statistic
"""

def occurrences(l):
    d_occurrences = {}
    for el in l:
        d_occurrences[el] = d_occurrences[el] + 1 if el in d_occurrences else 1
    return DICT.dsort(d_occurrences, by='value', reverse=True)

def describe(l, print_res=True, print_plot=True, colors=None, over=False, fsize=None, title=''):
    return DF.describe(pd.DataFrame(l, columns=['values']), 'values', 
                        print_res=print_res, print_plot=print_plot,
                        colors=colors, 
                        over=over, fsize=fsize, title=title)

#-----------------------------------------------------------------------------#

"""
Reshaping, flat 1-D and other dimension
"""

def flat(l):
    l = nparr(l)
    return l.reshape(-1)

def reshape(l, rows=None, cols=None, axis=0):
    l = nparr(l)
    reshaped_l = None
    if (rows==None and cols==None):
        UTILS.throw_msg('error', 'at least one of \'rows\' and \'cols\' attribute must be specified')
        return False
    elif (rows!=None and cols==None):
        if len(l)%rows != 0:
            UTILS.throw_msg('error', 'can\'t create ' + str(rows) + ' rows. List length: ' + str(len(l)) + ' reminder: ' + str(len(l)%rows))
            return False
        else:
            reshaped_l = reshape(l, rows, int(len(l)/rows))
    elif (rows==None and cols!=None):
        if len(l)%cols != 0:
            UTILS.throw_msg('error', 'can\'t create ' + str(cols) + ' columns. List length: ' + str(len(l)) + ' reminder: ' + str(len(l)%cols))
            return False
        else:
            reshaped_l = reshape(l, int(len(l)/cols), cols)
    else:
        reshaped_l = np.reshape(l, (rows, cols))
    return reshaped_l if axis==0 else np.transpose(reshaped_l)

#-----------------------------------------------------------------------------#

"""
Reverse
"""

def reverse(l):
    return l[::-1]

#-----------------------------------------------------------------------------#

"""
Replicate
"""

def replicate(element, n_times):
    return [element for n in range(n_times)]

#-----------------------------------------------------------------------------#

"""
Index of
"""

def index_of(l, element, mode='first'):
    if mode == 'first':
        return l.index(element) if element in l else -1
    elif mode == 'last':
        rl = l[::-1]
        return len(l)-rl.index(element) if element in rl else -1
    elif mode == 'all':
        idxs = [i for i,el in enumerate(l) if el==element]
        return idxs
    else:
        UTILS.throw_msg('error', 'Mode must be one of \'first\' (default), \'last\' or \'all\'')

#-----------------------------------------------------------------------------#

"""
Split list in sublists
"""

def split_sep(l, sep):
    lout = []
    while sep in l:
        idx = l.index(sep)
        if l[:idx] != []:
            lout.append(l[:idx])
        l = l[idx+1:]
    if l[:idx] != []:
        lout.append(l[:idx])
    return lout

def split_delta(l, delta):
    lout = []
    while l != []:
        lout.append(l[0:delta] if delta<=len(l) else l)
        l = l[delta:]
    return lout

#-----------------------------------------------------------------------------#

"""
Generic operation on each element
"""

def applyf(l, lambda_fun):
    return [lambda_fun(v) for v in l]

def applyzf(*l, lambda_fun):
    return [lambda_fun(*v) for v in zip(*l)]

def applycf(l, lambda_fun, chunk_dim=2):
    chunks = split_delta(l, chunk_dim)
    return list(flat([applyf(chunk, lambda_fun) for chunk in chunks]))

def applytf(l, lambda_fun, time_interval=0.10):
    return [UTILS.exec_interval(el, lambda_fun, time_interval) for el in l]

# TODO: Apply with sliding-window method
def applyswf(l, lambda_fun, swsize=5, mode=['exclude','extend_value','extend_0','extend_noise','circular'], extend_value=0):
    pass
    

#-----------------------------------------------------------------------------#

"""
Get element in index idx for list of lists
"""
def component(ll, comp_idx):
    return [l[comp_idx] for l in ll]

#-----------------------------------------------------------------------------#

"""
Set 
"""

def lset(l):
    s = list(set(l))
    return s if len(s)>1 else s[0]

#-----------------------------------------------------------------------------#

"""
Filter elements
"""

def lfilter(l, lambda_fun, mode='include'):
    if mode == 'include':       return [v for v in l if lambda_fun(v)]
    elif mode == 'exclude':     return [v for v in l if not lambda_fun(v)]
    else:                       return {'in': [v for v in l if lambda_fun(v)], 'out': [v for v in l if not lambda_fun(v)]}

#-----------------------------------------------------------------------------#

"""
Intersect
"""

def intersect(l1, l2):
    return [v for v in l1 if v in l2]

#-----------------------------------------------------------------------------#

"""
List transformation
"""

def normalize(l, exclude_01=False):
    minv = min(l)
    maxv = max(l)
    if exclude_01:
        if lfilter(l, lambda x: x>=0 and x<=1, mode='exclude') != []:
            UTILS.throw_msg('warning', 'exclude 0 and 1 does make sense only when all list values are in [0,1], this is not the case, standard procedure will be executed')
            minv = min(l)
            maxv = max(l)
        else:
            try:
                minv = min(lfilter(l, lambda x:x!=0))
                maxv = min(lfilter(l, lambda x:x!=1))
            except:
                UTILS.throw_msg('warning', 'Failed to exclude 0 and 1')
                minv = min(l)
                maxv = max(l)
    if minv==maxv:
        UTILS.throw_msg('error', 'can\'t normalize cause min value is the same of max value')
        return False
    else:
        return [((v-minv) / (maxv-minv)) if v!=0 and v!=1 else v for v in l]

def standardize(l):
    desc = describe(l, print_res=False, print_plot=False)
    return applyf(l, lambda x: (x-desc['mean'])/desc['std'])

def rescale(l, new_min, new_max):
    old_min = min(l)
    old_max = max(l)
    return [((((v - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min) for v in l]

def discretize(l, n, preprocess=None):
    l = normalize(l) if preprocess=='normalize' else standardize(l) if preprocess=='standardize' else l
    delta = (max(l)-min(l)) / n
    d = [min(l)]
    while(d[-1]<max(l)):
        d.append(d[-1]+delta)
    dis = []
    for v in l:
        i = 0
        while v > d[i+1]:
            i=i+1
        dis.append(i)
    return dis

def gamma_corr(l, gamma, preprocess=None):
    l = normalize(l) if preprocess=='normalize' else standardize(l) if preprocess=='standardize' else l
    maxv = max(l)
    return [math.pow((v / maxv), gamma) * maxv for v in l]

#-----------------------------------------------------------------------------#

###############################################################################