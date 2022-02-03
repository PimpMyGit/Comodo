###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'mongo_utils.py': MongoDB connection and management

###############################################################################

from re import escape
import pymongo

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT

###############################################################################

class MONGO:

    def __init__(self, db_name, host=MODULES._MONGO_LOCAL_HOST, port=MODULES._MONGO_LOCAL_PORT):
        self._client = pymongo.MongoClient(host, port)
        self._db = self._client[db_name]
        OBJECTS.add_object('_MONGO_' + db_name.upper() + ('_DB' if 'DB' not in db_name else ''), self)

    #-------------------------------------------------------------------------#

    def store_document(self, collection, document):
        try:
            self._db[collection].insert(document)
            UTILS.throw_msg('success', 'MongoDB inserted one documents into \'' + self._db.name + '.' + collection + '\'')
        except Exception as ex:
            UTILS.throw_msg('error', str(ex)+ + '\n' + 'Error on document: \n' + str(document))

    def store_documents(self, collection, documents):
        try:
            self._db[collection].insert_many(documents)
            UTILS.throw_msg('success', 'MongoDB inserted ' + str(len(documents)) + ' documents into \'' + self._db.name + '.' + collection + '\'')
        except Exception as ex:
            UTILS.throw_msg('error', str(ex))

    def upsert_documents(self, collection, documents, id_fields='_id'):
        try:
            for document in documents:
                self._db[collection].update_mDBy({id_fields:document[id_fields]}, {"$set":document}, upsert=True)
            UTILS.throw_msg('success', 'MongoDB upserted ' + str(len(documents)) + ' documents into \'' + self._db.name + '.' + collection + '\'')
        except Exception as ex:
            UTILS.throw_msg('error', str(ex) + '\n' + 'Error on document: \n' + str(document))

    #-------------------------------------------------------------------------#

###############################################################################