###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'comodo.py': Main utils functions container

###############################################################################

from .constants import *

from .BASE import general_utils as UTILS
from .BASE import list_utils as LIST
from .BASE import dict_utils as DICT

from .ANALYSIS import df_utils as DF
from .ANALYSIS import plot_utils as PLOT
from .ANALYSIS import ml_utils as ML
from .ANALYSIS import ts_utils as TS

from .DB import sparql_utils as SPQ
from .DB import mongo_utils as MDB
from .DB import arango_utils as ADB
from .DB import models_utils as MODELS
from .DB import manager_utils as MANAGERS

################################################################################

## How to import

# Store main dir as 'C:\..\Python\Python39\Lib\comodo'
# > from comodo.comodo import *

# Open a notebook dove cazzo vuoi, stai comodo.

################################################################################