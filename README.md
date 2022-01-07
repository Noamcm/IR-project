# IR-project
our IR course search engine

*search* function returns documents by using BM25 and sorting the posting lists by Tf in descending order. returns maximum 100 relevant documents.  
*search_body* function returns documents by using TF-IDF and cos-sim foreach document and query. returns maximum 100 relevant documents.  
*search_title* and *search_anchor* functions both return ALL relevant documents by binary ranking, ranked by descending order according to the amount of query terms that appear in the title/anchor.  
*get_pagerank* function returns page rank foreach wiki id's it receives. the page rank is computed by a page graph (which page has link to other pages).  
*get_pageview* function returns page views foreach wiki id's it receives.  
