#!/usr/bin/env python

import sys
import os
# curDir = os.path.dirname(__file__)#不知道为什么，windows下结果一样，但是linux下，不对！！！！
curDir = os.path.dirname(os.path.abspath(__file__)) #这样才对
parentDir = os.path.dirname(curDir)
sys.path.append(parentDir)
from preprocess import topics,docs
import json
import pprint
from elasticsearch import Elasticsearch
import requests

dataDir = os.path.join(parentDir, 'data')
docDir = os.path.join(dataDir, 'clinicaltrials_xml')
resultDir = os.path.join(dataDir, 'result.txt')

# simiModule: default,bm25 or vsm
def Body(simModule):
    initIndexBody = {
        "settings": {
            "number_of_shards": "2",
            "number_of_replicas": "0",
            # "similarity" : {          #setting bm25 https://elasticsearch.cn/book/elasticsearch_definitive_guide_2.x/changing-similarities.html
            #     "my_bm25": { 
            #         "type": "BM25",
            #         "b":    0 
            #     }
            # },
            "analysis": {
                "filter": {
                    "english_stop": {
                        "type":       "stop",
                        "stopwords":  "_english_"
                    },
                    "light_english_stemmer": {
                        "type":       "stemmer",
                        "language":   "light_english" 
                    },
                    "english_possessive_stemmer": {
                        "type":       "stemmer",
                        "language":   "possessive_english"
                    }
                },
                "analyzer": {
                    "my_analyzer": {
                        "type":  "standard",
                        "filter": [
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "light_english_stemmer", 
                            "asciifolding" 
                        ]
                    }
                }
            }
        },
        "mapping" : {
            "trial" :{
                "properties" : {
                    "nct_id" : {
                        "type" : "string",
                        "index" : "not_analyzed"
                    },
                    "brief_title" : {
                        "type" : "string",
                        "similarity": simModule
                    },
                    "official_title" : {
                        "type" : "string",
                        "similarity": simModule
                        # "boost" : 1
                    },
                    "brief_summary" : {
                        "type" : "string",
                        "similarity": simModule            # set similarity module, default is tf-idf
                    },
                    "study_type" :{
                        "type" : "string"
                    },
                    "primary_purpose" : {
                        "type" : "string"
                    },
                    "gender" : {
                        "type" : "string"
                    },
                    "minimum_age" : {
                        "type" : "integer"
                    },
                    "maximum_age" : {
                        "type" : "integer"
                    },
                    "healthy_volunteers" : {
                        "type" : "string"
                    }
                }
            }
        }
    }
    return initIndexBody
# connect to es ser ver
es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

def initIndex(indexName, Module):
    es.indices.create(index=indexName, body=Body(Module))

# add document to index, and set doc's is in index
def addIndex(indexName, id, docBody):
    es.index(index=indexName, doc_type='trial', id=id, body=docBody)

# test connection
req = requests.get(r'http://localhost:9200')
if req.status_code != 200:
    raise RuntimeError('connection failure')

# init the index


# convert each document to json object
dir0 = os.listdir(docDir)     #000 001 002...
for i in range(len(dir0)):
    path0 = os.path.join(docDir, dir0[i])       # clinicaltrials_xml/000
    dir1 = os.listdir(path0)                    # 00000,00001,00002...
    for j in range(len(dir1)):
        path1 = os.path.join(path0, dir1[j])    # clinicaltrials_xml/000/00000
        fileName = os.listdir(path1)            # NCT00000102.xml
        for k in range(len(fileName)):
            filePath = os.path.join(path1,fileName[k])
            rawDoc = docs.loadDoc(filePath)
            jsonDoc = rawDoc.toJsonObj()
            docId = rawDoc.getDocId()
            try:
                addIndex("clinicaltrials_bm25", docId, jsonDoc)         # add document to index
                addIndex("clinicaltrials_tfidf", docId, jsonDoc)
            except Exception as e:
                print(e)
        print('finish dir {}'.format(dir1[j]))
    print('finish dir {}'.format(dir0[i]))
print('Creat index successful!')