from elasticsearch import Elasticsearch

es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

es.delete(index='clinicaltrials', doc_type='trial', id="*")