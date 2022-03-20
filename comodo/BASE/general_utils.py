###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'general_utils.py': General utils functions

###############################################################################

import os
import re
import math
import time
import uuid
import json
import enum
import pickle
import inspect
import warnings
import xmltodict
import datetime

from enum import Enum

from termcolor import colored

from ..constants import *

from . import dict_utils as DICT
from . import list_utils as LIST

###############################################################################

"""
Throw message
"""

def throw_msg(category, message='', module=None, function=None, ex=None):
    category_color = {
        'done':'blue',
        'success':'green',
        'warning':'yellow',
        'error':'red'
    }
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0]) if module==None else module
    function = frame[3] if function==None else function
    if category.lower() in category_color:
        full_message = category.capitalize() + ' → ' + (module.__name__.upper() if module!=None else '') + ' in ' + function + '() → ' + message.capitalize()
        print(colored(full_message, category_color[category.lower()]))
        if type(ex) is Exception:
            raise ex
    else:
        throw_msg('error', 'message category muste be in [' + ', '.join(list(category_color.keys())) + ']')

#-----------------------------------------------------------------------------#

"""
Warning handler
"""

def warning_off():
    warnings.filterwarnings("ignore")

def warning_on():
    warnings.resetwarnings()

#-----------------------------------------------------------------------------#

def try_ex(lambda_do, ex, lambda_catch, lambda_final):
    try:
        lambda_do()
    except LIST.lvalues(ex):
        lambda_catch()
    if (lambda_final):
        lambda_final()

def try_pass(lambda_do, ):
    try:
        lambda_do()
    except:
        pass

def try_default(lambda_do, default=None, *lambda_params):
    try:
        return lambda_do(*lambda_params)
    except:
        return default
    
#-----------------------------------------------------------------------------#

"""
OS Paths
"""

def create_path(path):
    PATHS.create_path(path)
    throw_msg('Done', 'Path \'' + path + '\' has been created.')
    
#-----------------------------------------------------------------------------#

"""
Istruction sequence on element
"""

# safe on inplace method
def lambda_vsafe(element, lambda_fun):
    ret = lambda_fun(element)
    return ret if not ret is None else element

def lambda_seq(*params, lseq=[]):
    lseq_type = type(lseq)
    lseq = lseq if lseq_type is dict or lseq_type is list else LIST.lvalues(lseq)
    lout, lfuns = (list(lseq.keys()), list(lseq.values())) if type(lseq) is dict else ([n for n in range(len(lseq))], lseq)
    return {lo: lf(*params) for lo,lf in zip(lout,lfuns)} if lseq_type is dict else LIST.lvalue([lf(*params) for lf in lfuns])

def lambda_pipe(element, lpipe, first_no_args=False):
    lpipe = LIST.lvalues(lpipe)
    for i,lp in enumerate(lpipe):
        element = lp() if i == 0 and first_no_args else lambda_vsafe(element, lp)
    return element

#-----------------------------------------------------------------------------#

"""
Execute time interval
"""

def exec_interval(element, lambda_fun, time_interval=0.5):
    time.sleep(time_interval)
    return lambda_fun(element)

#-----------------------------------------------------------------------------#

"""
To Bool
"""

def int2bool(int_v):
    return False if int_v < 1 else True

#-----------------------------------------------------------------------------#

"""
Base conversion
"""

def bin2dec(bin_v):
    return int(bin_v, 2)

def dec2bin(dec):
    return bin(dec)[2:]

def hex2dec(hex_v):
    return int(hex_v, 16)

def dec2hex(dec):
    return hex(dec)[2:]

#-----------------------------------------------------------------------------#

"""
Code Generators
"""

def make_uuid(to_hex='False'):
    return uuid.uuid4().hex if to_hex else str(uuid.uuid4())

#-----------------------------------------------------------------------------#

"""
Colors
"""

def rgb2hex(rgb_tuple):
    return '#' + dec2hex(rgb_tuple[0]) + dec2hex(rgb_tuple[1]) + dec2hex(rgb_tuple[2])

def hex2rgb(hex_value):
    hex_value = hex_value[1:] if hex_value[0]=='#' else hex_value
    return (hex2dec(hex_value[0:2]), hex2dec(hex_value[2:4]), hex2dec(hex_value[4:6]))

def color_samples(n_samples, palette='rainbow'):
    pass

#-----------------------------------------------------------------------------#

"""
XML to DICT
"""
def xml2dict(xml):
    return json.loads(json.dumps(xmltodict.parse(xml)))

#-----------------------------------------------------------------------------#

"""
If None
"""

def if_none(obj, value):
    return value if obj is None else obj

def do_if(args, lambda_do, if_clause):
    if if_clause:
        lambda_do(*LIST.lvalues(args))

#-----------------------------------------------------------------------------#

"""
Date and time spilt components
"""

def datetime_components(timestamp, scheme='dd-MM-YYYY hh:mm:ss', dt_split=' ', d_split='-', t_split=':'):
    comps = timestamp.split(dt_split)
    s_comps = scheme.split(dt_split)
    d_comps = date_components(comps[0 if 'dd' in s_comps[0] else 1], s_comps[0 if 'dd' in s_comps[0] else 1], d_split=d_split)
    t_comps = time_components(comps[0 if 'hh' in s_comps[0] else 1], s_comps[0 if 'hh' in s_comps[0] else 1], t_split=t_split)
    dt_comps = DICT.merge(d_comps, t_comps) 
    return dt_comps

def date_components(date, scheme='dd-MM-YYYY', d_split='-'):
    comps = date.split(d_split)
    s_comps = scheme.split(d_split)
    d_comps = { 'DD': comps[s_comps.index('dd')],
                'MM': comps[s_comps.index('MM')],
                'YYYY': comps[s_comps.index('YYYY')] }
    return d_comps

def time_components(time, scheme='hh:mm:ss', t_split=':'):
        comps = time.split(t_split)
        s_comps = scheme.split(t_split)
        t_comps = { 'hh': comps[s_comps.index('hh')],
                    'mm': comps[s_comps.index('mm')],
                    'ss': comps[s_comps.index('ss')] }
        return t_comps
    
def today_date(scheme="%Y-%m-%d", as_str=False):
    return to_datetime(datetime_to_str(datetime.date.today(), scheme=scheme), scheme=scheme, as_str=as_str)

def date_weekday(str_date, scheme='DD-MM-YYYY', d_split='-'):
    dc = date_components(str_date, scheme=scheme, d_split=d_split)
    return datetime.date(int(dc['YYYY']), int(dc['MM']), int(dc['DD'])).weekday()

def to_datetime(str_date, scheme="%Y-%m-%d", as_str=False):
    date = datetime.datetime.strptime(str_date, scheme)
    return date if not as_str else datetime_to_str(date, scheme=as_str if type(as_str) is str else "%Y-%m-%d")

def datetime_to_str(date, scheme="%Y-%m-%d"):
    return date.strftime(scheme)

def is_date_valid(str_date, schema='%Y-%m-%d'):
    try:
        to_datetime(str_date, scheme=schema)
    except ValueError:
        return False
    return True

def generate_dates(anni, mesi=[], giorni=[], to_date_schema='%Y-%m-%d', as_str=False):
    anni = LIST.lvalues(anni)
    mesi = LIST.lvalues(mesi) if mesi!=[] else list(range(1,13))
    giorni = LIST.lvalues(giorni) if giorni!=[] else list(range(1,32))
    periodi = [str(anno) for anno in anni]
    periodi = LIST.flat([LIST.applyf(mesi, lambda mese: periodo + str(mese).rjust(2,'0')) for periodo in periodi])
    periodi = LIST.flat([LIST.applyf(giorni, lambda giorno: periodo + str(giorno).rjust(2,'0')) for periodo in periodi])
    periodi = LIST.lfilter(list(periodi), lambda d: is_date_valid(d, schema="%Y%m%d"))
    periodi = [to_datetime(periodo, scheme="%Y%m%d") for periodo in periodi]
    periodi_str = [datetime_to_str(periodo, scheme=to_date_schema) for periodo in periodi]
    if as_str:
        periodi =  periodi_str
    else:
        periodi = [to_datetime(periodo, scheme=to_date_schema) for periodo in periodi_str]
    return list(set(periodi))

#-----------------------------------------------------------------------------#

"""
Pickle save and load
"""

def pkl_save(filepath, obj, under_dir = PATHS._BASE_DIR, overwrite=True):
    levels = filepath.split('/')
    filename = levels[-1] + ('.pkl' if levels[-1][-4:]!='.pkl' else '')
    curr_dir = under_dir if under_dir!=None else PATHS._CW_DIR
    os.chdir(curr_dir)
    for level in levels[:-1]:
        if level not in os.listdir():
            os.mkdir(level)
        curr_dir += '/'+level
        os.chdir(curr_dir)
    if filename in os.listdir():
        if overwrite:
            throw_msg('warning', 'file \'' + filename + '\' will be overwrited')
        else:
            throw_msg('error', 'file \'' + filename + '\' already present, maybe you want to set overwrite=True')
            return False
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    os.chdir(PATHS._CW_DIR)
    throw_msg('done', 'object saved in ' + curr_dir.replace('/','\\') + '\\' + filename)
    return curr_dir.replace('/','\\') + '\\' + filename

def pkl_load(filepath, under_dir = PATHS._BASE_DIR):
    levels = filepath.split('/')
    filename = levels[-1] + ('.pkl' if levels[-1][-4:]!='.pkl' else '')
    curr_dir = under_dir if under_dir!=None else PATHS._CW_DIR
    os.chdir(curr_dir)
    for level in levels[:-1]:
        if level not in os.listdir():
            throw_msg('error', 'filepath not valid: directory \'' + level + '\' does not exists in \'' + os.getcwd() + '\'')
            return False
        else:
            curr_dir += '/'+level
            os.chdir(curr_dir)
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    os.chdir(PATHS._CW_DIR)
    return obj

#-----------------------------------------------------------------------------#

"""
Activation and other generic functions
"""

def bstep(x):
    return 0 if x < 0 else 1

def relu(x):
    return 0 if x < 0 else x

def logistic(x):
    return 1 / (1 + exp(-x))

def log(x, base=math.e):
    return math.log(x, base)

def exp(x, base=math.e):
    return math.pow(base, x)

def root(x, idx):
    return math.pow(x, 1/idx)

#-----------------------------------------------------------------------------#

"""
Index of (String)
"""

def index_of(s, element, mode='first'):
    if mode == 'first':
        return s.index(element) if element in s else -1
    elif mode == 'last':
        rs = s[::-1]
        return len(s)-rs.index(element)-1 if element in rs else -1
    elif mode == 'all':
        idxs = []
        delta = 0
        while element in s:
            idx = s.index(element)
            idxs.append(idx+delta)
            delta = delta + idx+len(element)
            s = s[idx+len(element):]
        return idxs
    else:
        UTILS.throw_msg('error', 'Mode must be one of \'first\' (default), \'last\' or \'all\'')

"""
Replace multiple chars/substrings
"""

def str_replace(s, rmap):
    for k,v in rmap.items():
        s = s.replace(k,str(v))
    return s

"""
Remove multiple space and start/end spaces
"""

def one_space(s):
    if s.replace(' ','') != '':
        s = re.sub(' +', ' ', s)
        if s[0] == ' ':
            s = s[1:]
        if s[len(s)-1] == ' ':
            s = s[:-1]
    return s

"""
Start / End with
"""

def start_with(s, start):
    return s[ : len(start)] == start

def end_with(s, end):
    return s[-len(end) : ] == end

def byte_to_str(byte, encoding="utf-8"):
    return str(byte)[2:-1] if encoding==None else byte.decode(encoding)

def str_to_df(s, line_sep='\n', sep=',', headers=True):
    data = [line.split(sep) for line in s.split(line_sep)]
    return pd.DataFrame(data[1:], columns=data[0]) if headers else pd.DataFrame(data)
    
def str_format(s, args):
    return str_replace(s, {'{'+str(ia)+'}':arg for ia,arg in enumerate(LIST.lvalues(args))})

#-----------------------------------------------------------------------------#

"""
Types handling and generating
"""

def En(values, name='BaseEnum'):
    values = values if type(values) is dict else {value: i for i,value in enumerate(values)}
    return Enum(name, values)

#-----------------------------------------------------------------------------#

"""
Input
"""

def ask_input(message, input_type=str, check=None, mandatory=False, default=None):
    valid = False
    while not valid:
        inp = one_space(input(message))
        if inp == '' and not mandatory:
            valid = True
            inp = default
        else:
            try:
                if input_type is str:
                    pass
                if input_type is int:
                    inp = int(inp)
                if input_type is list:
                    inp = [one_space(element) for element in inp.split(',')]
                if type(input_type) is enum.EnumMeta:
                    inp = input_type[inp]
                valid = True
                if not check is None:
                    valid = check(inp)
            except Exception as ex:
                valid = False
    return inp
    # if not check is None:
    #     valid = False
    #     while(not valid):
    #         try:
    #             valid = check(inp)
    #             valid = valid if mandatory else True
    #         except Exception as ex:
    #             valid = False if mandatory else True
    #             use_default = True if valid else False
    #         if not valid:
    #             inp = one_space(input(message))
    return inp if not use_default else default
                 



#     except Exception as ex:

#     while(not check(inp)):


###############################################################################