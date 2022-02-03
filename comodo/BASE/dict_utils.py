###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'list_utils.py': List utils functions

###############################################################################

from . import general_utils as UTILS

###############################################################################

"""
Pretty print
"""

def pretty(d, indent=0, keys=[], mode='exclude'):
    for key, value in d.items():
        if (key not in keys) if mode=='exclude' else (key in keys):
            print('\t' * indent + str(key) + ': ' + ('{' if type(value) is dict else ''))
            if type(value) is dict:
                pretty(value, indent+1, keys=keys, mode=mode)
                print('\t' * indent + '}')
            else:
                print('\t' * (indent+1) + str(value))

#----------------------------------------------------------------------------#

"""
Access by dot notation
"""

def dot_key(d, dk_str):
    keys = dk_str.split('.')
    return d[keys[0]] if len(keys)==0 else dot_key(d[keys[0]], '.'.join(keys[1:]))

#----------------------------------------------------------------------------#

"""
Sort functions
"""

def dsort(d, by='key', reverse=False):
    return sort_key(d, reverse=reverse) if by=='key' else sort_value(d, reverse=reverse)

def sort_key(d, reverse=False):
    return {key: d[key] for key in sorted(d.keys(), reverse=reverse)}

def sort_value(d, reverse=True):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

#----------------------------------------------------------------------------#

"""
Merge function
"""

def merge(d1, d2, keep='left'):
    d = {}
    for k,v in (d1.items() if keep=='right' else d2.items()):
        d[k] = v
    for k,v in (d2.items() if keep=='right' else d1.items()):
        if k in (d1.items() if keep=='right' else d2.items()):
            d[k] = v if keep=='right' else d[k] 
        else:
            d[k] = v
    return d
    
#-----------------------------------------------------------------------------#

"""
Dictionary subset and filter
"""

def subdict(d, keys, mode='include', warn=False):
    dout = d.copy()
    keys = [keys] if type(keys) is str else keys
    din = {}
    for key in keys:
        if key in dout:
            din[key] = d[key]
            dout.pop(key, None)
        else:
            if warn:
                UTILS.throw_msg('warning', 'Key \'' + key + '\' skipped because it is not present in dict')
    return din if mode=='include' else dout 

def dfilter(d, lambda_fun, by='key'):
    return filter_key(d, lambda_fun) if by=='key' else filter_value(d, lambda_fun)

def filter_key(d, lambda_fun):
    return {k: d[k] for k in d if lambda_fun(k)}

def filter_value(d, lambda_fun):
    return {k: d[k] for k in d if lambda_fun(d[k])}

#-----------------------------------------------------------------------------#

"""
Replace keys
"""

def replace(d, key_values, upsert=True):
    for key,value in key_values.items():
        d[key]=value
    return d

#-----------------------------------------------------------------------------#

"""
Rename keys
"""

def rename(d, rename_dict):
    return {(k if k not in rename_dict else rename_dict[k]): v for k,v in d.items()}

#-----------------------------------------------------------------------------#

"""
Generic operations on values
"""

def apply(d, lambda_fun, by='value', mode='single'):
    return apply_key(d, lambda_fun, mode=mode) if by == 'key' else apply_value(d, lambda_fun, mode=mode)

def apply_key(d, lambda_fun, mode='single'):
    if mode == 'list':
        tmp = lambda_fun(list(d.keys()))
        return {tmp[i]: d[k] for i,k in enumerate(d.keys())}
    else:
        return {lambda_fun(k): d[k] for k in d.keys()}

def apply_value(d, lambda_fun, mode='single'):
    if mode == 'list':
        tmp = lambda_fun(list(d.values()))
        return {k:tmp[i] for i,k in enumerate(d.keys())}
    else:
        return {k:lambda_fun(d[k]) for k in d.keys()}

#-----------------------------------------------------------------------------#

###############################################################################