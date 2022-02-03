###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'models_utils.py': A generic model class 

###############################################################################

import re
from typing import List
from ..constants import *

from . import models_utils as MODELS
from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST
from ..ANALYSIS import df_utils as DF

###############################################################################

class Base_Manager:

    def __init__(self, db, model=MODELS.Base_Model, params=None):
        self._mng_db = db
        self._model = model
        if type(params) is dict:
            for k,v in params.items():
                self._add_param(k,v)

    def _add_param(self, k, v):
        self.__dict__[k]=v


    # da eliminare
    def _make_models(self, list_dict, model=None):
        model = self._model if model == None else model
        return [model(obj) for obj in list_dict]

###############################################################################

class Sparql_Manager (Base_Manager):

    def __init__ (self, spq, model=MODELS.Base_Model, params=None):
        super().__init__(db=spq, model=model, params=params)
    
    #-------------------------------------------------------------------------#

    """
    Query to sparql endpoint → return a dataframe in list of dict format
    """

    def _get(self, q, dict_params={}, to_chunk=False, chunk_dim=100, to_list_dict=True):
        spq_out = None
        if to_chunk:
            spq_out = self._mng_db.chunk_query(q, dict_params=dict_params, chunk_dim=chunk_dim, format=True, parse_out=True, to_list_dict=to_list_dict)
        else:
            spq_out = self._mng_db.query(q if not dict_params else q(dict_params), format=True, parse_out=True, to_list_dict=to_list_dict)
        return spq_out
    
###############################################################################

class Arango_Manager (Base_Manager):

    def __init__ (self, adb, model=MODELS.Arango_Doc_Model, params=None):
        super().__init__(db=adb, model=model, params=params)
    
    """
    Store to ArangoDB
    """
    
    # region - Store document models 

    def store_models(self, collection_name, models, upsert=False):
        LIST.applyf(models, lambda model: model._build())
        self._mng_db._insert_documents(collection_name, [model._to_json() for model in models], upsert=upsert)
        return models

    def store_doc_from_df(self, df_models, collection_name, model_type=MODELS.Arango_Doc_Model, columns=None, group_columns=None, upsert=True):
        models = DF.list_dict(df_models, cols=columns, group_cols=group_columns)
        models = [model_type(model) for model in models]
        return self.store_models(collection_name=collection_name, models=models, upsert=upsert)

    # endregion

    # region - Store edge models 
    
    def store_link_by_df(self, df_edges, from_column, to_column, from_collection_name, to_collection_name, edge_model=MODELS.Arango_Edge_Model, edge_columns_params=None, edge_collection_name=None, add_uuid_key=False):
        map_from_models = {from_id: self.load_by_key(from_collection_name, from_id.replace('/','..')) for from_id in list(set(df_edges[from_column]))}
        map_to_models = {to_id: self.load_by_key(to_collection_name, to_id.replace('/','..')) for to_id in list(set(df_edges[to_column]))}
        edges_models = LIST.lfilter(df_edges.apply(lambda row: edge_model(map_from_models[row[from_column]], map_to_models[row[to_column]], (dict(row[LIST.lfilter(list(df_edges.columns), lambda col: col!=from_column and col!=to_column)]) if edge_columns_params==None else dict(row[LIST.lvalues(edge_columns_params)])), add_uuid_key=add_uuid_key) if map_from_models[row[from_column]]!=None and map_to_models[row[to_column]]!=None else None, axis=1), lambda doc: doc != None)
        collection = self._mng_db._collection((edge_collection_name if edge_collection_name else from_collection_name + '__' + to_collection_name), collection_type=MODULES._ARANGO_COLLECTION_TYPE_EDGE)
        return self.store_links(collection_name=collection.name, edges=edges_models) 

    def store_links(self, collection_name, edges):
        LIST.applyf(edges, lambda edge: edge._build())
        self._mng_db._insert_edges(collection_name, [edge._to_json() for edge in edges])
        return edges

    # endregion

    #-------------------------------------------------------------------------#

    """
    Load from ArangoDB
    """

    # region - Loading models

    def load_models(self, collection_name, model=MODELS.Arango_Doc_Model, returns=[]):
        docs = self._mng_db._return_documents(collection_name, returns=LIST.lvalues(returns))
        models = [model(doc) for doc in docs]
        LIST.applyf(models, lambda m: m._load())
        return models

    def load_by_key(self, collection_name, _key, model=MODELS.Arango_Doc_Model):
        doc = self._mng_db._return_unique(collection_name, _key, by='_key')
        return model(doc) if not doc is None else None

    # endregion

###############################################################################

class Mongo_Manager (Base_Manager):

    def __init__ (self, mdb, model=MODELS.Mongo_Model, params=None):
        super().__init__(db=mdb, model=model, params=params)

    #-------------------------------------------------------------------------#

    """
    Store Models
    """
    
    def store_models(self, collection_name, models, upsert=False):
        LIST.applyf(models, lambda model: model._build())
        if upsert:
            self._mng_db.upsert_documents(collection_name, [model._to_json() for model in models])
        else:            
            self._mng_db.store_documents(collection_name, [model._to_json() for model in models])
        return models

###############################################################################

