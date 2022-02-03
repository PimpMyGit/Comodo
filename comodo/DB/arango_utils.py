###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'arango_utils.py': ArangoDB connection and management

###############################################################################

from re import escape
from pyArango.connection import *
from pyArango.theExceptions import *

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST

###############################################################################

class ARANGO:

    def __init__(self, db_name, arangoURL = MODULES._ARANGO_LOCAL_URL, username = MODULES._ARANGO_ROOT_USER['username'], password = MODULES._ARANGO_ROOT_USER['password'], verify = True):
        self._client = Connection(arangoURL=arangoURL, username=username, password=password, verify=verify)
        if not self._client.hasDatabase(db_name):
            self._client.createDatabase(db_name)
        self._db = self._client[db_name]
        OBJECTS.add_object(self._global_db_name(), self)

    def _global_db_name(self):
        return '_ARANGO_' + self._db.name.upper() + ('_DB' if 'DB' not in self._db.name else '')
       

    #-------------------------------------------------------------------------#

    """
    Collection
    """
    # region - Collections

    def _create_collection(self, collection_name, collection_type=None):
        if not self._db.hasCollection(collection_name):
            self._db.createCollection(name=collection_name, type=UTILS.if_none(collection_type, MODULES._ARANGO_COLLECTION_TYPE_DOCUMENT))
            MODULES.add_property(self._global_db_name() + '_' + collection_name.upper() + '_COLLECTION', collection_name)
            return self._get_collection(collection_name)
        else:
            UTILS.throw_msg('warning', 'Collection \''+collection_name+'\' already exists in db \''+self._db.name+'\'')      
            return None      

    def _get_collection(self, collection_name):
        return self._db.collections[collection_name] if self._db.hasCollection(collection_name) else None

    def _collection(self, collection_name, collection_type=None):
        collection = self._get_collection(collection_name)
        return collection if collection != None else self._create_collection(collection_name, collection_type)

    # endregion

    #-------------------------------------------------------------------------#

    """
    Documents and Links
    """
    
    def _build_documents(self, collection_name, documents):
        docs = LIST.lvalues(documents)
        collection = self._collection(collection_name)
        return LIST.lvalue([collection.createDocument(doc) for doc in docs])

    def _build_edges(self, collection_name, edges):
        edges = LIST.lvalues(edges)
        collection = self._collection(collection_name, collection_type=MODULES._ARANGO_COLLECTION_TYPE_EDGE)
        return LIST.lvalue([collection.createDocument(edge) for edge in edges])

    def _insert_document(self, collection_name, document, upsert=False):
        try:
            if upsert:
                tud = self._doc_by_key(collection_name, document['_key'])
                document = self._update_doc_body(tud) if tud!=None else document
            document.save()
        except Exception as ex:
            UTILS.throw_msg('Warning', 'Skipped duplicate insertion of doc: ' + str(document['_key']) + ':\n' + str(document))

    def _insert_documents(self, collection_name, documents, upsert=False):
        if upsert:
            self._upsert_documents(collection_name, documents)
        else:
            docs = self._build_documents(collection_name, documents)
            LIST.applyf(docs, lambda d: self._insert_document(collection_name, d, upsert=upsert))

    def _insert_edges(self, collection_name, edges): 
        edges = self._build_edges(collection_name, edges)
        for edge in edges:
            try:
                edge.save()
            except CreationError as cex:
                UTILS.throw_msg('Warning', 'Creation error of doc: ' + str(edge['_key']) + ':\n' + str(edge))

    def _update_document(self, to_update_document, document):
        self._update_doc_body(to_update_document, document).save()

    def _update_documents(self, collection_name, to_update_documents, documents):
        exists = [self._doc_by_key(tud['_key']) for tud in to_update_documents]
        if None in exists:
            UTILS.throw_msg('error', 'Not all the documents passed already exist. Use upsert_documents instaed.' )
        else:
            documents = self._build_documents(collection_name, documents)
            LIST.applyzf(exists, documents, lambda_fun=lambda tud, doc: self._update_document(tud, doc))

    def _upsert_documents(self, collection_name, documents):
        docs = self._build_documents(collection_name, documents)
        exists = [self._doc_by_key(collection_name, tud['_key']) for tud in docs]
        LIST.applyzf(exists, docs, lambda_fun=lambda tud, doc: self._update_document(tud, doc) if tud!=None else self._insert_document(collection_name, doc))

    def _doc_by_key(self, collection_name, key):
        try:
            return self._collection(collection_name)[key]
        except DocumentNotFoundError:
            return None

    def _update_doc_body(self, to_update_doc, doc):
        updated_doc = DICT.merge(to_update_doc.getStore(), DICT.subdict(doc.getStore(), '_key', mode='exclude'), keep='right')
        to_update_doc.set(updated_doc)
        return to_update_doc

    def _return_unique(self, collection_name, unique, by='_key'):
        _aql = """
            FOR doc IN """ + collection_name + """
                FILTER doc.""" + by + """ == '""" + unique + """'
            RETURN doc
        """
        results = self._db.AQLQuery(_aql, rawResults=True)
        return results[0] if len(results)>0 else None

    def _return_documents(self, collection_name, filters={}, returns=[]):
        docs = [doc.getStore() for doc in self._get_collection(collection_name).fetchAll()]
        if len(returns) > 0:
            if '!_key' not in returns and '_key' not in returns:
                returns.append('_key')
            docs = [DICT.subdict(doc, returns) for doc in docs]
        return docs

    #-------------------------------------------------------------------------#

###############################################################################