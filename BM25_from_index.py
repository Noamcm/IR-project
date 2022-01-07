import math
from itertools import chain
import time
import numpy as np

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.


def get_candidate_documents(query_to_search,index,words,pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. 
    
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score. 
    """
    candidates = []    
    for term in np.unique(query_to_search):
        if term in words:        
            current_list = (pls[words.index(term)])                
            candidates += current_list    
    return np.unique(candidates)


class BM25_from_index:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self,index,DL,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values())/self.N
        self.words, self.pls = zip(*self.index.posting_lists_iter())        

    def calc_idf(self,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """        
        idf = {}        
        for term in list_of_tokens:            
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass                             
        return idf
        

    def search(self, query_terms,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query. 
        This function return a list of scores for the query as the following:
               value: a ranked list of pairs (doc_id, score) in the length of N.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        bm25dict = {}
        self.idf=self.calc_idf(query_terms)
        candidates = get_candidate_documents(query_terms,self.index,self.words,self.pls)
        docid_score = []
        for candidate in candidates:
            docid_score.append((candidate,self._score(query_terms,candidate)))
        docid_score_sorted = sorted([(doc_id,score) for doc_id, score in docid_score], key = lambda x: x[1],reverse=True)[:N]
        return docid_score_sorted
        #raise NotImplementedError()

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """        
        score = 0.0        
        doc_len = self.DL.get(str(doc_id),0)       
             
        for term in query:
            if term in self.index.term_total.keys():                
                term_frequencies = dict(self.pls[self.words.index(term)])                
                if doc_id in term_frequencies.keys():            
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score