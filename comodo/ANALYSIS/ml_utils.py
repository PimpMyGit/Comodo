###############################################################################

# Author: Tommaso Redaelli
# Year: 2021
# Description: A collection of utils generic function for data analysis

# This file 'ml_utils.py': Machine Learning utils functions

###############################################################################

import math
import numpy as np
import pandas as pd
import scipy as scipy
import seaborn as sns
import matplotlib.pyplot as plt

from fastcluster import linkage
from scipy.spatial.distance import pdist, squareform, cosine

from sklearn.cluster import DBSCAN

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import list_utils as LIST

from . import plot_utils as PLT

###############################################################################

"""
Word 2 Vec
"""

def w2v(sentences, model='SG', vector_size=300, min_count=10, get_bigrams=True, min_count_bigrams=10):
    if get_bigrams:
        UTILS.throw_msg('Done', 'Getting bigrams ..')
        sentences = generate_bi_grams(sentences, min_count=min_count_bigrams, threshold=0.5, progress_per=1000, scoring='npmi')
        UTILS.throw_msg('Success', 'Done bigrams.')

    UTILS.throw_msg('Done', 'Building ' + model + ' model ...')
    w2v_model = Word2Vec(sentences, min_count=min_count, vector_size=vector_size, sg=(0 if model=='CBOW' else 1))
    UTILS.throw_msg('Success', 'Model completed.')
    return w2v_model

def generate_bi_grams(sentences, **kwargs):
    phrases = Phrases(sentences, **kwargs)
    phrases_model = Phraser(phrases)
    return [phrases_model[sent] for sent in sentences]

# --------------------------------------------------------------------------- #

def wvec(wv_model, w):
    return wv_model.wv[w]

def wnorm(wv_model, w):
    vec = wv_model.wv[w] if type(w) is str else w
    return math.sqrt(sum([math.pow(v,2) for v in vec]))

def wsim(wv_model, pos, neg=[], topn=10, thresh=0, comp=-1):
    pos = [pos] if type(pos) is str else pos
    neg = [neg] if type(neg) is str else neg
    sim0 = list(wv_model.wv.most_similar(positive=pos, negative=neg, topn=topn))
    if thresh > 0:
        sim0 = [s for s in sim0 if s[1] >= thresh]
    if comp != -1:
        sim0 = [s[0] if comp==0 else s[1] for s in sim0]
    return sim0

def vsim(wv_model, vector, topn=10, comp=-1):
    sim = wv_model.wv.similar_by_vector(vector, topn=topn)
    if comp == 0:
        return [s[0] for s in sim]
    elif comp == 1:
        return [s[1] for s in sim]  
    else:
        return sim

def wanal(wv_model, str_anal, topn=10, thresh=0):
    p1 = str_anal[:str_anal.index('=')]
    p2 = str_anal[str_anal.index('=')+1:]
    neg = p1[:p1.index(':')].replace(' ', '')
    pos1 = p1[p1.index(':')+1:].replace(' ', '')
    pos2 = p2[:p2.index(':')].replace(' ', '')
    return wsim(wv_model, [pos1, pos2], neg, topn=topn, thresh=thresh)

def wdist(wv_model, w1, w2):
    w1 = wv_model.wv[w1] if type(w1) is str else w1
    w2 = wv_model.wv[w2] if type(w2) is str else w2
    return 1-scipy.spatial.distance.cosine(w1, w2)

def nwdist(wv_model, ws1, ws2):
    ws1 = [w for w in ws1 if w in wv_model.wv.key_to_index]
    ws2 = [w for w in ws2 if w in wv_model.wv.key_to_index]
    return wv_model.wv.n_similarity(ws1,ws2)

def wchoesion(wv_model, words):
    wsum = 0
    for w1 in words:
        for w2 in words:
            wsum = wsum + ((1-wdist(wv_model, w1, w2)) if w1!=w2 else 0)
    return (wsum / (math.pow(len(words),2)-len(words))) if (math.pow(len(words),2)-len(words))!=0 else 0

###############################################################################

"""
Distance Matrix
"""

def dist_matrix(values1, values2=[], lambda_fun=lambda x,y: abs(x-y), transform=lambda mat: LIST.normalize(mat), print_plot=True, fsize=None, title=''):
    matrix=[]
    two_sets = True
    if len(values2) == 0:
        two_sets = False
        values2 = values1
    for v1 in values1:
        row = []
        for v2 in values2:
            row.append(lambda_fun(v1, v2))
        matrix.append(row)
    if type(transform) is type(lambda x:x):
        vlen = len(matrix[0])
        flat_matrix = transform(list(np.array(matrix).reshape(-1)))
        matrix = list(LIST.reshape(flat_matrix, cols=vlen))
    if print_plot:
        PLT.heatmap(LIST.nparr(matrix), fsize=fsize, title=title)
    return {'matrix': matrix, 'values': {'rows':values1, 'cols':values2} if two_sets else values1}

def sort_dist_matrix(dist_matrix, method='complete', initial_values=[], transform=lambda mat: LIST.normalize(mat), print_plot=True, fsize=None, title=''):
    if type(transform) is type(lambda x:x):
        vlen = len(dist_matrix[0])
        flat_matrix = transform(LIST.flat(dist_matrix))
        dist_matrix = list(np.array(flat_matrix).reshape(int(len(flat_matrix)/vlen),vlen))
    if int(dist_matrix[0][0]) == 1:
        dist_matrix = [[(1-v) for v in row] for row in dist_matrix]
    if type(dist_matrix) is not np.array:
        dist_matrix = np.array([np.array(row) for row in dist_matrix])
    out = compute_serial_matrix(dist_matrix, method)
    if print_plot:
        PLT.heatmap(out['sorted_dist_matrix'], fsize=fsize, title=title)
    if initial_values != []:
        out['values'] = [initial_values[i] for i in out['res_sorted']]
    return out

def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def compute_serial_matrix(dist_mat, method='complete'):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_sorted is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return {'sorted_dist_matrix':seriated_dist, 'res_sorted':res_order, 'res_linkage':res_linkage}

#-----------------------------------------------------------------------------#

""" Clustering DB-SCAN """

def dm_cluster(dist_matrix, elements, eps='auto-75', min_samples=2):
    if type(eps) is str and 'auto' in eps:
        flat = list(np.reshape(dist_matrix,-1))
        eps = ((sum(flat)-math.sqrt(len(flat))) / (len(flat)-math.sqrt(len(flat)))) - (0 if eps=='auto-50' else np.std(flat))
        eps = eps if eps > 0 else 0.1
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)
    out = {
        'core_sample_indices': clustering.core_sample_indices_,
        'components': clustering.components_,
        'labels': list(clustering.labels_),
    }
    cluster = {c:{'words': [], 'score':0} for c in list(set(out['labels']))}
    for i,l in enumerate(out['labels']):
        try:
            cluster[l]['words'].append(elements[i])
        except:
            print(i,l,elements)
    # for c in list(cluster.keys()):
    #     rfreq = len(cluster[c]['words'])/len(elements)
    #     rfreq_noout = len(cluster[c]['words'])/(len(elements)-(0 if -1 not in cluster else len(cluster[-1]['words']))) if (len(elements)-(0 if -1 not in cluster else len(cluster[-1]['words'])))>0 else 0
    #     cho_score = wchoesion(cluster[c]['words'])
    #     cluster[c]['score'] = {'rfreq': rfreq, 'rfreq_noout':rfreq_noout, 'choesion':cho_score}
    out['cluster'] = cluster
    return out

###############################################################################