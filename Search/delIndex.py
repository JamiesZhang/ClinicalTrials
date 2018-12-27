from elasticsearch import Elasticsearch

es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

es.indices.delete(index='clinicaltrials_bm25')
es.indices.delete(index='clinicaltrials_tfidf')