

from inverted_index_gcp import *
import hashlib
import builtins
import pandas as pd
import pickle


from BM25_from_index import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# stemmer = PorterStemmer()
all_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


# ************************** CREATE INDEXES *******************************

# !mkdir text title anchor
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def doc_count(text, doc_id):
    tokens = tokenize(text)
    countTokens = OrderedDict(Counter(tokens))
    countTokens_len = builtins.sum(countTokens.values())
    return doc_id, countTokens_len


# def improved_tokenize(text):
#     list_of_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if
#                       token.group() not in all_stopwords]  ##is it work?
#     return list_of_tokens


# ************************** SEARCH BODY BY COSIM *******************************

def get_top_n(sim_dict, dct_title):
    lst = sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                 reverse=True)[:100]
    return [(int(x[0]), dct_title.get(x[0], 0)) for x in lst]


def calculate_tfidf(query, idx_text, DL_text, bucket, dct_title,directory):
    clean_query = tokenize(query)
    sim_dct = {}
    total_terms_in_query = dict(Counter(clean_query))
    NFquery = 1 / ((builtins.sum([x * x for x in total_terms_in_query.values()])) ** 0.5)
    for term in clean_query:
        pls = []
        try:
            pls = idx_text.read_posting_list(term, bucket,directory)
        except:
            pls = []
        query_freq = total_terms_in_query[term]
        for doc_id, doc_freq in pls:  # if there's no posting list, it won't calculate and stay 0
            sim_dct[doc_id] = sim_dct.get(doc_id, 0) + doc_freq * query_freq
    for doc_id in sim_dct.keys():
        NFdoc = 1 / (DL_text[doc_id])
        sim_dct[doc_id] = sim_dct.get(doc_id, 0) * NFquery * NFdoc
    return get_top_n(sim_dct, dct_title)


# ************************** END SEARCH BODY BY COSIM *******************************


# ************************** SEARCH TITLE AND ANCHOR BY BINARY  *******************************

def search_by_binary(query, index, bucket,dct_title,directory):
    clean_query = tokenize(query)
    candidates = []
    for term in np.unique(clean_query):
        candidates = candidates + list(map(lambda x: x[0], index.read_posting_list(term, bucket,directory)))
    in_order = list(map(lambda x: (x[0], dct_title.get(x[0], 0)), (Counter(candidates)).most_common()))
    return in_order


# ************************** END SEARCH TITLE AND ANCHOR BY BINARY  *******************************

# ************************** GET PAGERANK/PAGEVIEW  *******************************


def get_page_stats(wiki_ids, path):
    with open(path, 'rb') as f:  # read in the dictionary from disk
        wid2pr = pickle.loads(f.read())
    result = []
    for doc_id in wiki_ids:
        result.append(wid2pr.get(doc_id, 0))
    return result


# ************************** END GET PAGERANK/PAGEVIEW  *******************************


# ************************** SEARCH  *******************************

def search_back(query, bucket_name, idx_title, idx_text, idx_anchor, dct_title, DL_title, DL_text, DL_anchor):
    clean_query = tokenize(query)
    bm25title = BM25_from_index(idx_title, DL_title)
    bm25text = BM25_from_index(idx_text, DL_text)
    bm25anchor = BM25_from_index(idx_anchor, DL_anchor)
    title_candidates = bm25title.search(clean_query, bucket_name, 'title')
    text_candidates = bm25text.search(clean_query, bucket_name, 'text')
    anchor_candidates = bm25anchor.search(clean_query, bucket_name, 'anchor')
    best_text_title = merge_results(title_candidates, text_candidates, anchor_candidates, title_weight=0.1,
                                    text_weight=0.5, anchor_weight=0.2, pagerank_weight=0.2, N=100)
    return list(map(lambda tup: (int(tup[0]), dct_title[tup[0]]), best_text_title))


def merge_results(title_scores, body_scores, anchor_scores, title_weight=0.25, text_weight=0.25, anchor_weight=0.25,
                  pagerank_weight=0.25, N=3):
    title_dict = dict(title_scores)
    body_dict = dict(body_scores)
    anchor_dict = dict(anchor_scores)
    docs_id_for_query = set(list(title_dict.keys()) + list(body_dict.keys()) + list(anchor_dict.keys()))
    allscores = []
    i = 0
    pagerank = get_page_stats(docs_id_for_query, "./pagerank.pkl")
    for docid in docs_id_for_query:
        allscores.append((docid, title_dict.get(docid, 0) * title_weight + body_dict.get(docid,
                                                                                         0) * text_weight - anchor_dict.get(
            docid, 0) * anchor_weight + pagerank[i] * pagerank_weight))
        i += 1
    score_sorted = sorted(allscores, key=lambda x: x[1], reverse=True)[:N]
    return score_sorted

# ************************** END SEARCH  *******************************
