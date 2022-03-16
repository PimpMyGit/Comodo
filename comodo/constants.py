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

import html
import re

from nltk.tokenize import word_tokenize, sent_tokenize

class _LAMBDAS:
    """
    Costant execution functios / routines
    """

    _LAMBDAS_DICT = {

        '_TXT_TEXT_NORMALIZATION_STEPS': {
            'LOWER':            lambda s:   s.lower(),                                                                  # Minuscolo grazie, microfono fibra, ah
            'DECODE_HTML':      lambda s:   html.unescape(s),                                                           # Caratteri speciali
            'NO_URL':           lambda s:   re.sub(r'https?://\S+|www.\.\S+', '', s),                                   # No urls
            'NO_EMOJI':         lambda s:   UTILS.one_space(re.compile("["                                              # No emoticon
                                                        u"\U0001F600-\U0001F64F" 
                                                        u"\U0001F300-\U0001F5FF" 
                                                        u"\U0001F680-\U0001F6FF" 
                                                        u"\U0001F1E0-\U0001F1FF" 
                                                        u"\U00002500-\U00002BEF" 
                                                        u"\U00002702-\U000027B0"
                                                        u"\U00002702-\U000027B0"
                                                        u"\U000024C2-\U0001F251"
                                                        u"\U0001f926-\U0001f937"
                                                        u"\U00010000-\U0010ffff"
                                                        u"\u2640-\u2642"
                                                        u"\u2600-\u2B55"
                                                        u"\u200d"
                                                        u"\u23cf"
                                                        u"\u23e9"
                                                        u"\u231a"
                                                        u"\ufe0f"
                                                        u"\u3030"
                                                        "]+", flags=re.UNICODE).sub(r' ', s)),
            'NO_ESCAPE':        lambda s:   UTILS.one_space(UTILS.str_replace(s, {'\n':' ', '\r':' ', '\t':' '})),      # Cursori vari
            'NO_PUNCTUATION':   lambda s:   UTILS.one_space(UTILS.str_replace(s, {                                      # Punteggiatura
                                                                                    '!'     :   ' ',
                                                                                    '"'     :   ' ',
                                                                                    '#'     :   ' ',
                                                                                    '$'     :   ' ',
                                                                                    '%'     :   ' ',
                                                                                    '&'     :   ' ',
                                                                                    '\''    :   '\'',
                                                                                    '('     :   ' ',
                                                                                    ')'     :   ' ',
                                                                                    '*'     :   ' ',
                                                                                    '+'     :   ' ',
                                                                                    ','     :   ' ',
                                                                                    '’'     :   ' ',
                                                                                    '”'     :   ' ',

                                                                                    ' - '   :   ' ',
                                                                                    '- '    :   ' ', 
                                                                                    ' -'    :   ' ',
                                                                                    '-'     :   '_',

                                                                                    '/'     :   ' ',
                                                                                    ':'     :   ' ',
                                                                                    ';'     :   ' ',
                                                                                    '<'     :   ' ',
                                                                                    '='     :   ' ',
                                                                                    '>'     :   ' ',
                                                                                    '?'     :   ' ',
                                                                                    '['     :   ' ',
                                                                                    '\''    :   ' ',
                                                                                    ']'     :   ' ',
                                                                                    '^'     :   ' ',
                                                                                    '_'     :   ' ',
                                                                                    '`'     :   ' ',
                                                                                    '{'     :   ' ',
                                                                                    '|'     :   ' ',
                                                                                    '}'     :   ' ',
                                                                                    '~'     :   ' ',
                                                                                })),
            'NO_NUMBERS':       lambda s:   UTILS.one_space(re.sub(r'(\d\d*\.?)+', ' ', s)),                            # Numeri
            'NO_ELLIPSIS':      lambda s:   UTILS.one_space(re.sub(r'\.{2,}', '. ', s)),                                # Puntini di sospensione
        },

        '_TXT_TEXT_TOKENIZATION_STEPS': {
            'SENTENCE':     lambda text:    sent_tokenize(text),
            'WORD':         lambda sentences:   [word_tokenize(sentence) if type(sentence) is str and len(sentence)>0 else None for sentence in sentences]
        }

    }

    def __init__(self):
        self._configure()

    def _configure(self):
        for k,v in self._LAMBDAS_DICT.items():
            self.add_lambda(k,v)

    def add_lambda(self, name, value):
        self.__dict__[name] = value

#-----------------------------------------------------------------------------#

LAMBDAS = _LAMBDAS()

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
        if 'LAMBDAS' in dict_config:
            [LAMBDAS.add_lambda(name, path) for name, path in dict_config['LAMBDAS'].items()]
        
        UTILS.throw_msg('done', 'Comodo\'s costants correctly configured')
    except Exception as ex:
        UTILS.throw_msg('error', str(ex))

###############################################################################