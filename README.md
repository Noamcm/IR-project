# IR-project
A good search engine has three main features:
Quick search - yields results for certain queries within a few seconds.
Relevant results - yields results related to a query entered.
Sorted results - The first results that the engine will display will be most related to the query and will be sorted in descending order from most related to less related.
  
The search engine we created in this project searches for results using a number of methods:  
*search_body* Returns up to 100 results for the query by calculating tf-idf and cosine similarity in the body of the document.  
*search_title* and *search_anchor* functions both returns all relevant results for the query using boolean search on document titles / links. Sort the results from the document in its title / links the most words from the query to the document in its title / links appeared as few words as possible from the query. (Does not include documents that do not contain but a word from the query).  
*get_pagerank* function returns page rank foreach wiki id's it receives. the page rank is computed by a page graph (which page has link to other pages).  
*get_pageview* function returns page views foreach wiki id's it receives.  
  
  
*search* Returns up to 100 results for the query as follows:  
-Checks what the relevant documents are according to the posting lists of the query words for the titles, text and anchor.  
-For each of these documents checks what their pageRank value is.  
-Calculates BM25 value for title, text and anchor separately.  
-Merges the BM25 and pageRank values to a single value for each document.  
-Returns the relevant documents sorted in descending order according to the weighted value.
