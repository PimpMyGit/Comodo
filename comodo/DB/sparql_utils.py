###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'sparql_utils.py': Costants for different application

###############################################################################

import time
import numpy as np
import pandas as pd
import SPARQLWrapper as spq

from urllib.error import HTTPError

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST

from ..ANALYSIS import df_utils as DF

###############################################################################

class SPARQL():

    def __init__(self, endpoint=MODULES._SPQ_DBPEDIA_ENDPOINT, return_format=MODULES._SPQ_RETURN_FORMAT_JSON):
        self.endpoint = endpoint
        self.return_format = return_format
        self._sparql = spq.SPARQLWrapper(endpoint, return_format)
        OBJECTS.add_object('_SPARQL_' + endpoint[endpoint.index('//')+2:endpoint.rindex('.')].replace('.','_').upper(), self)

    #-------------------------------------------------------------------------#

    """
    Query formatting & printing
    """

    def no_comments(self, q):
        q = q.split('\n')
        qs = [l for l in q if '#' not in l]
        return '\n'.join(qs)

    def inline(self, q):
        q = q.replace('    ','') # tab
        q = q.replace('\n',' ')  # new line
        q = q[1:] if q[0] == ' ' else q
        return q

    def set_prefix(self, q):
        q = UTILS.str_replace(q, MODULES._SPQ_PREFIX_URI_MAP)
        return q

    def format_query(self, q):
        q = self.no_comments(q)
        q = self.inline(q)
        q = self.set_prefix(q)
        return q

    def print_query(self, q):
        print(self.no_comments(q))

    #-------------------------------------------------------------------------#

    """
    Get query type
    """

    def query_type(self, q, format=False):
        q = self.format_query(q) if format else q
        qtype = q[:q.index(' ')]
        return qtype.lower()

    #-------------------------------------------------------------------------#

    """
    Query handling
    """
    
    def select_query(self, qout, parse_out=True, to_list_dict=False):
        out = self.parse_sparql_df(qout, to_list_dict=to_list_dict) if parse_out else qout
        return out

    def describe_query(self, qout, parse_out=True, to_list_dict=False):
        return self.parse_sparql_df(qout, parse_out=parse_out, to_list_dict=to_list_dict)

    def ask_query(self, qout, parse_out=True):
        out = qout['boolean'] if parse_out else qout
        return out

    """
    Parse query result to pandas.DataFrame
    """
    def parse_sparql_df(self, out, to_list_dict=False):
        if self.return_format == MODULES._SPQ_RETURN_FORMAT_JSON:
            keys = out['head']['vars']
            values = out['results']['bindings']
            entries = [{k : v[k]['value'] for k in keys} for v in values]
            df = pd.DataFrame(entries, columns=keys)
            return df

        elif self.return_format == MODULES._SPQ_RETURN_FORMAT_XML:
            out = UTILS.xml2dict(out.toxml())
            keys = [var['@name'] for var in out['sparql']['head']['variable']] if type(out['sparql']['head']['variable']) is list else [out['sparql']['head']['variable']['@name']]
            if 'result' in out['sparql']['results']:
                values = [val['binding'] for val in out['sparql']['results']['result']] if type(out['sparql']['results']['result']) is list else [out['sparql']['results']['result']['binding']]
                dict_values = []
                for value in values:
                    dict_value = {}
                    for val in (value if type(value) is list else [value]):
                        k = val['@name']
                        vk = list(DICT.subdict(val, '@name', mode='exclude').keys())[0]
                        if type(val[vk]) is dict:
                            v = val[vk]['#text']
                        else:
                            v = val[vk]
                        dict_value[k]=v
                    dict_values.append(dict_value)
                df = pd.DataFrame(dict_values, columns=keys)
                if df.shape[0]==0:
                    UTILS.throw_msg('warning', 'Empty result.')
                elif df.shape[0] >= MODULES._SPQ_MAX_ROWS_RETRIEVE:
                    UTILS.throw_msg('warning', 'Retrieved max rows ('+ str(MODULES._SPQ_MAX_ROWS_RETRIEVE) + '). You should try with smaller chunks.' )

                if to_list_dict:
                    return DF.list_dict(df)
                elif len(list(df.columns))==1:
                    return list(df.iloc[:,0])
                else:
                    return df
            else:
                return None

    #-----------------------------------------------------------------------------#

    """
    Send Query
    """

    def set_query(self, q, max_try=5):
        qout = None
        try:
            self._sparql.setQuery(q)
            qout = self._sparql.query().convert()
            return qout
        except HTTPError as http_err:
            if max_try > 0 and http_err.code == 503:
                UTILS.throw_msg('warning', 'HTTP 503, Service unavailable, retry again..')
                time.sleep(5)
                return self.set_query(q, max_try-1)
            else:
                raise http_err

    def query(self, q, format=True, parse_out=True, to_list_dict=False):
        q = self.format_query(q) if format else q

        qout = self.set_query(q)

        qtype = self.query_type(q)
        if qtype == 'select':
            return self.select_query(qout, parse_out, to_list_dict=to_list_dict)
        elif qtype == 'describe':
            return self.describe_query(qout, parse_out, to_list_dict=to_list_dict)
        elif qtype == 'ask':
            return self.ask_query(qout, parse_out)
        else:
            UTILS.throw_msg('warning', 'Query type \'' + qtype + '\' not recognized')
            return qout

    def chunk_query(self, q, dict_params, chunk_dim=100, format=True, parse_out=True, to_list_dict=False):
        chunk_pname = dict_params['to_chunk']
        chunks = LIST.split_delta(dict_params[chunk_pname], chunk_dim)
        chunks = LIST.flat(chunks) if chunk_dim==1 else chunks
        df = None
        chunk = chunks[0]
        df = self.query(q(DICT.replace(dict_params, {chunk_pname: chunk})), to_list_dict=False)
        if len(chunks)>0:
            for cidx, chunk in enumerate(chunks[1:]):
                chunk_df = self.query(q(DICT.replace(dict_params, {chunk_pname: chunk})), to_list_dict=False)
                df = DF.append(df, chunk_df) if type(df)!=type(None) and type(chunk_df)!=type(None) else df if type(df)!=type(None) else chunk_df if type(chunk_df)!=type(None) else None
                if cidx%10==0 or cidx==len(chunks)-2:
                    UTILS.throw_msg('done', 'Done chunck ' + str(cidx) + ' of ' + str(len(chunks)))
        out = df if not to_list_dict else DF.list_dict(df)
        return out

    #-----------------------------------------------------------------------------#

###############################################################################