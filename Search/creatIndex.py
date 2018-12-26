#!/usr/bin/env python

import sys
import os
curDir = os.path.dirname(__file__)
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

initIndexBody = {
    "settings": {
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
                    "similarity": "default"
                },
                "official_title" : {
                    "type" : "string",
                    "similarity": "default"
                },
                "brief_summary" : {
                    "type" : "string",
                    "similarity": "BM25"            # set similarity model, default is tf-idf
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

# connect to es ser ver
es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

# add document to index, and set doc's is in index
def addIndex(docBody, id):
    es.index(index='clinicaltrials', doc_type='trial', id=id, body=docBody)

# test connection
req = requests.get(r'http://localhost:9200')
if req.status_code != 200:
    raise RuntimeError('connection failure')

# init the index
es.index(index='clinicaltrials', doc_type='trial', id=id, body=initIndexBody)

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
                addIndex(jsonDoc, docId)         # add document to index
            except Exception as e:
                print(e)
