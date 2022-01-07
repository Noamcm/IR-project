# from collections import Counter, OrderedDict
# from graphframes import *
from inverted_index_colab import *
# from itertools import islice, count, groupby
from nltk.corpus import stopwords
# from nltk.stem.porter import *
# from operator import itemgetter
# from pathlib import Path
# from pyspark import SparkContext, SparkConf
# from pyspark.ml.feature import Tokenizer, RegexTokenizer
# from pyspark.sql import *
# from pyspark.sql import SQLContext
# from pyspark.sql.functions import *
# from time import time
# from timeit import timeit
# import os
# import nltk
# import hashlib
# import itertools
# import sys
# import pyspark
import builtins
import math
import numpy as np
import pandas as pd
import pickle
import re
from nltk.stem.porter import *
from BM25_from_index import *


stemmer = PorterStemmer()
all_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
# class search_backend:

def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


# ************************** SEARCH BODY BY COSIM *******************************
def get_posting_gen(index):
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def generate_query_tfidf_vector(query_to_search, index, DL):
    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros(total_vocab_size)
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df.get(token, 0)
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls, DL):
    candidates = {}
    N = len(DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                normlized_tfidf.append((doc_id, (freq / DL[doc_id]) * math.log(N / index.df[term], 10)))

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls, DL):
    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls, DL)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf
    return D


def cosine_similarity(D, Q):
    cosSim_dict = {}
    docs = D.to_numpy()
    docid = 0
    for doc in docs:
        mone = np.dot(doc, Q)
        mechane = np.linalg.norm(doc) * np.linalg.norm(Q)
        cosSim = mone / mechane
        cosSim_dict[D.index[docid]] = cosSim
        docid += 1
    return cosSim_dict


def get_top_n(sim_dict):
    lst= sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                  reverse=True)[:100]
    id_title_dct = InvertedIndex.read_index('title', 'index_title_dct')
    return [(int(x[0]), id_title_dct.get(x[0], 0)) for x in lst]


def get_topN_score_for_queries(clean_query, index):
    words, pls = get_posting_gen(index)
    DL = InvertedIndex.read_index('text', 'index_text_dct')
    Q = generate_query_tfidf_vector(clean_query, index, DL)
    D = generate_document_tfidf_matrix(clean_query, index, words, pls, DL)
    cosSim_dict = cosine_similarity(D, Q)
    topN = get_top_n(cosSim_dict)
    return topN


# ************************** END SEARCH BODY BY COSIM *******************************


# ************************** SEARCH TITLE AND ANCHOE BY BINARY  *******************************

def search_by_binary(clean_query, index):
    words, pls = get_posting_gen(index)
    candidates = []
    for term in np.unique(clean_query):
        if term in words:
            candidates = candidates + list(map(lambda x: x[0], pls[words.index(term)]))

    id_title_dct = InvertedIndex.read_index('title', 'index_title_dct')
    in_order = list(map(lambda x: (x[0], id_title_dct.get(x[0], 0)), (Counter(candidates)).most_common()))
    return in_order


# ************************** END SEARCH TITLE AND ANCHOE BY BINARY  *******************************

# ************************** GET PAGE RANK  *******************************

# def generate_graph(pages):
#     source = pages.map(lambda x: (x[0],))
#     target = pages.flatMap(lambda x: map(lambda y: (y[0],), x[1]))
#     vertices = (source.union(target)).distinct()
#     edges = (pages.flatMap(lambda x: list(map(lambda y: (x[0], y[0]), x[1])))).distinct()
#     return edges, vertices
#
#
# def calculate_page_rank(pages):
#     edges, vertices=generate_graph(pages)
#     edgesDF = edges.toDF(['src', 'dst']).repartition(4, 'src')
#     verticesDF = vertices.toDF(['id']).repartition(4, 'id')
#     g = GraphFrame(verticesDF, edgesDF)
#     pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
#     pr = pr_results.vertices.select("id", "pagerank")
#     pr = pr.sort(col('pagerank').desc())
#     return pr

def get_page_stats(wiki_ids, path):
    with open(path, 'rb') as f:  # read in the dictionary from disk
        wid2pr = pickle.loads(f.read())
    result = []
    for id in wiki_ids:
        result.append(wid2pr.get(id, 0))
    return result
# ************************** END GET PAGE RANK  *******************************



# ************************** SEARCH  *******************************


def improved_tokenize(text):
    list_of_tokens =  [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords] ##is it work?
    return list_of_tokens


def search_back(clean_query):
    idx_title = InvertedIndex.read_index('search_title', 'index_title')
    idx_text = InvertedIndex.read_index('search_text', 'index_text')
    # topNtitle = get_topN_score_for_queries(clean_query,idx_title)
    # topNtext = get_topN_score_for_queries(clean_query,idx_text)
    dct_title = InvertedIndex.read_index('search_title', 'index_title_dct')
    DL_title = dict(map(lambda tup: (tup[0], len(tup[1].split(' '))), dct_title.items()))
    DL_text = InvertedIndex.read_index('search_text', 'index_text_dct')
    bm25title = BM25_from_index(idx_title, DL_title)
    bm25text = BM25_from_index(idx_text, DL_text)
    title_candidates = bm25title.search(clean_query)
    text_candidates = bm25text.search(clean_query)
    best_text_title = merge_results(title_candidates, text_candidates, N=100)
    return list(map(lambda tup: (int(tup[0]), dct_title[tup[0]]), best_text_title))


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
    # YOUR CODE HERE
    title_dict = dict(title_scores)
    body_dict = dict(body_scores)
    docs_id_for_query = set(list(title_dict.keys()) + list(body_dict.keys()))
    allscores = []
    for docid in docs_id_for_query:
        allscores.append((docid, title_dict.get(docid, 0) * title_weight + body_dict.get(docid, 0) * text_weight))
    score_sorted = sorted(allscores, key=lambda x: x[1], reverse=True)[:N]
    return score_sorted


# ************************** END SEARCH  *******************************