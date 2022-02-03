###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'models_utils.py': A generic model class 

###############################################################################

import numpy as np

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT

###############################################################################

class Base_Model:

    def __init__(self, params):
        if type(params) is dict and params!={}:
            for k,v in params.items():
                self._add_param(k,v)

    def _add_param(self, key, value):
        self.__dict__[key] = value

    def _dict(self, key):
        return self.__dict__[self.__dict__[key]]

    def _to_json(self):
        d = {}
        for key,value in self.__dict__.items():
            try:
                d[key] = self._to_dict_format(value)
            except Exception as ex:
                UTILS.throw_msg('error', str(ex))
        return d

    def _to_dict_format(self, value):
        try:
            if isinstance(value, Base_Model):
                return value._to_json()
            elif isinstance(value, list):
                return [self._to_dict_format(v) for v in value]
            else:
                return value
        except Exception as ex:
            UTILS.throw_msg('error', str(ex))
        
    def _to_csv(self):
        csv_line = str(self._id) + ';'
        for param in DICT.sort_key(DICT.subdict(self.__dict__, '_id', 'exclude')).values():
            csv_line = str(param) + ';'
        return csv_line[:-1]

#-----------------------------------------------------------------------------#

class Mongo_Model(Base_Model):
    
    def __init__(self, params, id_name='_id'):
        if id_name in params:
            self._id = params[id_name]
            super().__init__(DICT.subdict(params, id_name, 'exclude'))
        else:
            super().__init__(params)

#-----------------------------------------------------------------------------#

class Arango_Doc_Model(Base_Model):
    
    def __init__(self, params={}, key_name='_key'):
        if key_name in params:
            self._key = params[key_name]
            super().__init__(DICT.subdict(params, key_name, 'exclude'))
        else:
            super().__init__(params)
    
    def _format_key(self):
        self._key = self._key.replace('/','..')

    def _unformat_key(self):
        self._key = self._key.replace('..','/')

    def _format_date(self, date, scheme="%Y-%m-%d"):
        try:
            date = date[:8]
            date = UTILS.to_datetime(date[:4]+'-'+date[4:6]+'-'+date[6:], scheme=scheme, as_str=True)
            return date
        except Exception as ex:
            return None

    def _build(self):
        self._format_key()
        self._check_nan()

    def _load(self):
        self._unformat_key()

    def _check_nan(self):
        for k,v in self.__dict__.items():
            try:
                if np.isnan(v):
                    self.__dict__[k] = None
            except TypeError as tex:
                pass
            except Exception as ex:
                raise ex
                

class Arango_Edge_Model(Arango_Doc_Model):

    def __init__(self, _from_model, _to_model, params={}, key_name='_key', add_uuid_key=False):
        self._from = _from_model._id
        self._to = _to_model._id
        if key_name in params:
            self._key = params[key_name]
            super().__init__(DICT.subdict(params, key_name, 'exclude'))
        else:
            self._key = _from_model._key + '@' + _to_model._key + ('@' + UTILS.make_uuid(to_hex='True') if add_uuid_key else '')
            super().__init__(params)

###############################################################################