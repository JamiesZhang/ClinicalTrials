from elasticsearch import Elasticsearch

es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

if not es.indices.exists(index='clinicaltrials_bm25'):
    print("The index \"clinicaltrials_bm25\" does not exist, you don't need delete it")
else:
    es.indices.delete(index='clinicaltrials_bm25')
    print('clinicaltrials_bm25 delete successful！')

print()

if not es.indices.exists(index='clinicaltrials_bm25'):
    print("The index \"clinicaltrials_tfidf\" does not exist, you don't need delete it")
else:
    es.indices.delete(index='clinicaltrials_tfidf')
    print('clinicaltrials_tfidf delete successful！')