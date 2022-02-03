###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'costants.py': Costants for different application

###############################################################################

import os

###############################################################################

# All paths we need, this folder + working directories and files

class _PATHS:
    """
    Paths class, basically a key-value dict
    """
    _PATHS_DICT = {

    # Directories

        '_CW_DIR'       :   os.getcwd(),
        '_BASE_DIR'     :   os.getcwd(),

    # Others ...

    }

    def __init__(self):
        self._configure(base_dir = False)

    def _configure(self, base_dir=True):
        for k,v in self._PATHS_DICT.items():
            self.add_path(k,v)
    
    def add_path(self, name, path):
        self.__dict__[name] = path
        if not os.path.isdir(path):
            self.create_path(path)

    def create_path(self, path):
        path = path.split('//')[:-1] if len(path.split('//')) > 1 else path[:-2] if path[-2:]=='//' else path
        p = ''
        for branch in path:
            p += branch + '//'
            try:
                os.chdir(p)
            except OSError or FileNotFoundError:
                os.mkdir(p)
        os.chdir(PATHS._BASE_DIR)

#-----------------------------------------------------------------------------#

PATHS = _PATHS()

###############################################################################

# All module costants / conficuration settings

import ssl
import pandas as pd
import seaborn as sns
import SPARQLWrapper as spq

from pyArango import consts as adb_consts

class _MODULES:
    """
    Modules' costants
    """

    _PROPERTY_DICT = {
        
    # Seaborn

        '_SNS_20x20_FIG_SIZE'   :   (20,20),
        '_SNS_4x3_FIG_SIZE'     :   (16,12),    # default
        '_SNS_30x10_FIG_SIZE'   :   (30,10),
        '_SNS_7x7_FIG_SIZE'     :   (7,7),
        '_SNS_2x2_FIG_SIZE'     :   (2,2),
        '_SNS_COLOR_PALETTE'    :   {1:'pastel', 2:'bright', 3:'colorblind', 4:'deep'}[1],

    # SSL

        '_SSL_HTTP_CONTEXT'     :   ssl._create_unverified_context,
    
    # SPARQL Wrapper

        '_SPQ_MAX_ROWS_RETRIEVE'    :   10000,

        '_SPQ_DBPEDIA_ENDPOINT'     :   'http://dbpedia.org/sparql',
        '_SPQ_RETURN_FORMAT_JSON'   :   spq.JSON,
        '_SPQ_RETURN_FORMAT_XML'    :   spq.XML,

        '_SPQ_PREFIX_URI_MAP'   :   {
                                        'dcterms:alternative'   :   '<http://purl.org/dc/terms/alternative>',
                                    },

    # MONGO DB

        '_MONGO_TEST_DB_NAME'   :   'TestKomodoDB',
    
        '_MONGO_LOCAL_HOST'     :   'localhost',
        '_MONGO_LOCAL_PORT'     :   27017,

    # ARANGO Local Client

        '_ARANGO_TEST_DB_NAME'  :   'TestKomodoDB',

        '_ARANGO_LOCAL_URL'     :   'http://127.0.0.1:8529',
        '_ARANGO_ROOT_USER'     :   {
                                        'username': 'root', 
                                        'password': 'root'
                                    },

        '_ARANGO_COLLECTION_TYPE_DOCUMENT'  :   adb_consts.COLLECTION_DOCUMENT_TYPE,
        '_ARANGO_COLLECTION_TYPE_EDGE'      :   adb_consts.COLLECTION_EDGE_TYPE,
    
    # Others ...

    }

    def __init__(self):
        self._configure()

    def _configure(self):
        for k,v in self._PROPERTY_DICT.items():
            self.add_property(k,v)

    def add_property(self, name, value):
        self.__dict__[name] = value

    def has_property(self, name):
        return name in self.__dict__

    def set_sns_fsize(self, fsize):
        sns.set(rc = {'figure.figsize': fsize})

    def restore_sns_fsize(self):
        self.set_sns_fsize(self._SNS_4x3_FIG_SIZE)

#-----------------------------------------------------------------------------#

MODULES = _MODULES()

###############################################################################

class _OBJECTS:
    """
    Costant Objects
    """

    _OBJECTS_DICT = {

    # ...

    }

    def __init__(self):
        self._configure()

    def _configure(self):
        for k,v in self._OBJECTS_DICT.items():
            self.add_object(k,v)

    def add_object(self, name, value):
        self.__dict__[name] = value

#-----------------------------------------------------------------------------#

OBJECTS = _OBJECTS()

###############################################################################

from .BASE import general_utils as UTILS

"""
Register attributes and functions from dict '{ 'PATHS':{}, 'MODULES':{}, 'OBJECTS':{} }'
"""

def register_consts(dict_config):
    try:
        if 'PATHS' in dict_config:
            [PATHS.add_path(name, path) for name, path in dict_config['PATHS'].items()]
        if 'MODULES' in dict_config:
            [MODULES.add_property(name, path) for name, path in dict_config['MODULES'].items()]
        if 'OBJECTS' in dict_config:
            [OBJECTS.add_object(name, path) for name, path in dict_config['OBJECTS'].items()]
        
        UTILS.throw_msg('done', 'Comodo\'s costants correctly configured')
    except Exception as ex:
        UTILS.throw_msg('error', str(ex))

###############################################################################