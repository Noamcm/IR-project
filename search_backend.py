from inverted_index_gcp import *
import hashlib
import builtins
import pandas as pd
import pickle
from BM25_from_index import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

all_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


# ************************** TEXT MANIPULATION *******************************

def _hash(s):
    # this function recieves text and returns it's hash value (number)
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def tokenize(text):
    # this function recieves text (a query) and returns a list of tokens after removing stopwords and irrelevant characters
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


# ************************** SEARCH BODY BY COSIM *******************************

def get_top_n(sim_dict, dct_title):
    # this function gets sim_dict: a dictionary with scores, key: doc-id and value: tf-idf score
    # dct_title: a dictionary which contains for each key:doc-id , value: name-of-doc
    # it returns 100 docs that has the top scores in the simdict
    lst = sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                 reverse=True)[:100]
    return [(int(x[0]), dct_title.get(x[0], 0)) for x in lst]


def calculate_tfidf(query, idx_text, DL_text, bucket, dct_title,directory):
    # query the client gave to the server
    # idx_text index for the body of the docs
    # DL_text:a dictionary: key: doc-id , value: doc-length
    # bucket: out bucket name
    # dct_title: a dictionary which contains for each key:doc-id , value: name-of-doc
    # directory: the relevant directory to search for bins (posting lists) 
    clean_query = tokenize(query)
    sim_dct = {}
    total_terms_in_query = dict(Counter(clean_query)) #for each term, count the number of appearances in the given query
    NFquery = 1 / ((builtins.sum([x * x for x in total_terms_in_query.values()])) ** 0.5) #normalization factor for the query
    for term in clean_query:
        pls = []
        try:
            pls = idx_text.read_posting_list(term, bucket,directory)
        except:
            pls = []
        query_freq = total_terms_in_query[term]
        for doc_id, doc_freq in pls:  # if there's no posting list, it won't calculate and stay 0
            sim_dct[doc_id] = sim_dct.get(doc_id, 0) + doc_freq * query_freq #sum all the calculations
    for doc_id in sim_dct.keys():
        NFdoc = 1 / (DL_text[doc_id]) #normalization factor for the current doc
        sim_dct[doc_id] = sim_dct.get(doc_id, 0) * NFquery * NFdoc #calculate the score for each doc
    return get_top_n(sim_dct, dct_title)


# ************************** SEARCH TITLE AND ANCHOR BY BINARY  *******************************

def search_by_binary(query, index, bucket,dct_title,directory):
    # query the client gave to the server
    # index: anchor/title index from the bucket, depends on the calling function
    # dct_title: a dictionary which contains for each key:doc-id , value: name-of-doc
    # directory: the relevant directory to search for bins (posting lists) 
    clean_query = tokenize(query)
    candidates = []
    for term in np.unique(clean_query):
        candidates = candidates + list(map(lambda x: x[0], index.read_posting_list(term, bucket,directory))) # concatenate all posting lists for the given terms
    in_order = list(map(lambda x: (x[0], dct_title.get(x[0], 0)), (Counter(candidates)).most_common())) # count the appearances for each doc_id, each appearance related to a specific term
    return in_order

# ************************** GET PAGERANK/PAGEVIEW  *******************************

def get_page_stats(wiki_ids, path):
    #this method gets:
    #   wiki_ids: list of doc_ids
    #   path: a path for a file in the bucket
    #returns: a list of the values pagerank/pageview depands on the calling functions
    with open(path, 'rb') as f:  # read in the dictionary from disk
        wid2pr = pickle.loads(f.read())
    result = []
    for doc_id in wiki_ids:
        result.append(wid2pr.get(doc_id, 0))
    return result

# ************************** SEARCH  *******************************

def search_back(query, bucket_name, idx_title, idx_text, idx_anchor, dct_title, DL_title, DL_text, DL_anchor):
    # this function recieves a query that the client entered, indexes for each part of the document (title,text,anchor) and dictionaries for their length
    # the function return the top 100 docs that related to the query according to BM25 calculation and pagerank.
    clean_query = tokenize(query)
    # create BM25 objects for each index
    bm25title = BM25_from_index(idx_title, DL_title)
    bm25text = BM25_from_index(idx_text, DL_text)
    bm25anchor = BM25_from_index(idx_anchor, DL_anchor)
    # calculate for each part of the document the top 2500 documents that relevant to the given query in the specific index
    title_candidates = bm25title.search(clean_query, bucket_name, 'title')
    text_candidates = bm25text.search(clean_query, bucket_name, 'text')
    anchor_candidates = bm25anchor.search(clean_query, bucket_name, 'anchor')
    # merge all the candidates and calculate the top 100 doc-id that relevant to the query
    best_text_title = merge_results(title_candidates, text_candidates, anchor_candidates, title_weight=0.1,
                                    text_weight=0.5, anchor_weight=0.2, pagerank_weight=0.2, N=100)
    return list(map(lambda tup: (int(tup[0]), dct_title[tup[0]]), best_text_title)) # returns a list, each element is a tuple in the format: (doc-id, title-name)


def merge_results(title_scores, body_scores, anchor_scores, title_weight=0.25, text_weight=0.25, anchor_weight=0.25,
                  pagerank_weight=0.25, N=3):
    # this function gets 3 lists of tuples, each element is a (doc-id, BM25 score) 
    # it merges all the candidates to one list and chooses the top 100 doc-id according to the scores and weights
    title_dict = dict(title_scores)
    body_dict = dict(body_scores)
    anchor_dict = dict(anchor_scores)
    docs_id_for_query = set(list(title_dict.keys()) + list(body_dict.keys()) + list(anchor_dict.keys())) # all doc-ids , each element is unique
    allscores = [] #each element will be: (doc-id, score-after-weights)
    i = 0
    pagerank = get_page_stats(docs_id_for_query, "./pagerank.pkl") #calculate the pagerank for each doc-id
    for docid in docs_id_for_query:
        allscores.append((docid, title_dict.get(docid, 0) * title_weight + body_dict.get(docid,
                                                                                         0) * text_weight - anchor_dict.get(
            docid, 0) * anchor_weight + pagerank[i] * pagerank_weight)) # the score for each doc-id
        i += 1
    score_sorted = sorted(allscores, key=lambda x: x[1], reverse=True)[:N]
    return score_sorted

# ************************** END SEARCH  *******************************
