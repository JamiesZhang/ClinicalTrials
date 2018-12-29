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
from elasticsearch.helpers import bulk
import requests

dataDir = os.path.join(parentDir, 'data')
docDir = os.path.join(dataDir, 'clinicaltrials_xml')
resultDir = os.path.join(dataDir, 'result.txt')

# simiModule: default,bm25 or vsm
def Body(simModule):
    initIndexBody = {
        "settings": {
            "number_of_shards": "4",
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

# high bulk
def gendata(indexName, path, dir):
    for dir3 in dir:
        dir4 = os.path.join(path, dir3)  #C:\Users\61759\Desktop\ClinicalTrials\data\clinicaltrials_xml\001\00100
        fileNames = os.listdir(dir4)
        for fileName in fileNames:
            filePath = os.path.join(dir4,fileName)
            rawDoc = docs.loadDoc(filePath)
            jsonDoc = rawDoc.toJsonObj()
            yield {
                "_index": indexName,
                "_type": "trial",
                "doc": jsonDoc,
            }
# def gendata(indexName, absPath, allFile):
#     for File in allFile:
#         filePath = os.path.join(absPath,File)
#         rawDoc = docs.loadDoc(filePath)
#         jsonDoc = rawDoc.toJsonObj()
#         yield {
#             "_index": indexName,
#             "_type": "trial",
#             "doc": jsonDoc,
#         }


# connect to es ser ver
es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

def initIndex(indexName, Module):
    if es.indices.exists(index=indexName):
        print("The index is already exists!!!")
        exit()
    else:
        es.indices.create(index=indexName, body=Body(Module))

# add document to index, and set doc's is in index
def addIndex(client, indexName, absPath, allFile):
    bulk(client, gendata(indexName, absPath, allFile))

# test connection
req = requests.get(r'http://localhost:9200')
if req.status_code != 200:
    raise RuntimeError('connection failure')

# init the index
initIndex("clinicaltrials_bm25",Body("BM25"))
initIndex("clinicaltrials_tfidf",Body("default"))

# convert each document to json object
dir0 = os.listdir(docDir)     #000 001 002...
for d in dir0:
    path0 = os.path.join(docDir, d)       # C:\Users\61759\Desktop\ClinicalTrials\data\clinicaltrials_xml\000
    dir1 = os.listdir(path0)
    bulk(es, gendata("clinicaltrials_bm25", path0, dir1))
    bulk(es, gendata("clinicaltrials_tfidf", path0, dir1))
    print('finish dir {}'.format(d))
print('Creat index successful!')

