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

from pprint import pprint

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec, LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary

import pyLDAvis
import pyLDAvis.gensim_models

from ..constants import *

from ..BASE import general_utils as UTILS
from ..BASE import dict_utils as DICT
from ..BASE import list_utils as LIST

from . import plot_utils as PLT

###############################################################################

"""
Text Preprocessing
"""

def txt_normalization(text, lambdas_steps=LAMBDAS._TXT_TEXT_NORMALIZATION_STEPS, exclude=None):
    if type(lambdas_steps) is dict and exclude:
        exclude = LIST.lvalues(exclude)
        lambdas_steps = DICT.filter_key(lambdas_steps, lambda step: step not in exclude)
    lambdas_steps = list(lambdas_steps.values()) if type(lambdas_steps) is dict else LIST.lvalues(lambdas_steps)
    lambdas_steps = LIST.lvalues(lambdas_steps)
    normalized_text = UTILS.lambda_pipe(text, lambdas_steps)
    return normalized_text

def txt_tokenization(text, lambdas_steps=LAMBDAS._TXT_TEXT_TOKENIZATION_STEPS, exclude=None):
    if type(lambdas_steps) is dict and exclude:
        exclude = LIST.lvalues(exclude)
        lambdas_steps = DICT.filter_key(lambdas_steps, lambda step: step not in exclude)
    lambdas_steps = list(lambdas_steps.values()) if type(lambdas_steps) is dict else LIST.lvalues(lambdas_steps)
    lambdas_steps = LIST.lvalues(lambdas_steps)
    tokenized_text = UTILS.lambda_pipe(text, lambdas_steps)
    return tokenized_text

###############################################################################

"""
TF-IDF
"""

def tfidf(document_words, round_decimals=2, min_df=0., max_df=1., use_df=True, **kwargs_tfidf):
    tv = TfidfVectorizer(min_df=min_df, max_df=max_df, use_idf=use_df, **kwargs_tfidf)
    tv_matrix = tv.fit_transform(document_words).toarray()
    tfidf_df = pd.DataFrame(np.round(tv_matrix, round_decimals), columns=tv.get_feature_names())
    return tfidf_df

###############################################################################

""" Topic-Modelling LDA """

def lda_topic_modelling(documents_words, num_topics, word_sep=' ', dict_filter_params={'no_below':5, 'no_above':0.5, 'keep_n':100000}, offset=2, random_state=100, update_every=0, passes=10, alpha='auto', eta='auto', per_word_topics=False, **kwargs_lda):
    docs_dictionary = Dictionary(LIST.applyf(documents_words, lambda s: s.split(word_sep) if type(s) is str else []))
    if dict_filter_params:
        docs_dictionary.filter_extremes(**dict_filter_params) 
    bow_corpus = [docs_dictionary.doc2bow(doc) for doc in LIST.applyf(documents_words, lambda s: s.split(word_sep) if type(s) is str else [])]
    lda_model = LdaModel(   bow_corpus,
                            id2word = docs_dictionary,
                            num_topics = num_topics,
                            offset = offset,
                            random_state = random_state,
                            update_every = update_every,
                            passes = passes,
                            alpha = alpha,
                            eta = eta,
                            per_word_topics = per_word_topics, 
                            **kwargs_lda    )
    out = {
        'lda_model': lda_model,
        'lda_dictionary': docs_dictionary,
        'lda_corpus': bow_corpus
    }
    return out

def lda_get_topics(lda_model, print_res=False):
    topics = lda_model.print_topics()
    if print_res:
        pprint(topics)
    return topics

def lda_plot(lda_model, lda_dictionary, lda_corpus):
    pyLDAvis.enable_notebook()
    lda_viz = pyLDAvis.gensim_models.prepare(lda_model, lda_dictionary, lda_corpus)
    return lda_viz

###############################################################################

"""
Word 2 Vec
"""

def w2v(sentences, model='SG', vector_size=300, min_count=10, get_bigrams=True, min_count_bigrams=10):
    if get_bigrams:
        sentences = generate_bi_grams(sentences, min_count=min_count_bigrams, threshold=0.5, progress_per=1000, scoring='npmi')
        UTILS.throw_msg('Success', 'Done bigrams.')

    w2v_model = Word2Vec(sentences, min_count=min_count, vector_size=vector_size, sg=(0 if model=='CBOW' else 1))
    UTILS.throw_msg('Success', 'Model completed.')
    return w2v_model

def get_w2v_vocab(w2v_model):
    return w2v_model.wv.key_to_index

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