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

# connect to es ser ver
es = Elasticsearch([{'host':'localhost', 'port' : 9200}])

# add document to index, and set doc's is in index
def addIndex(docBody, id):
    es.index(index='clinicaltrials', doc_type='trial', id=id, body=docBody)

# test connection
req = requests.get(r'http://localhost:9200')
if req.status_code != 200:
    raise RuntimeError('connection failure')

# load query(topics)
rawtopics = topics.loadRawTopics()

# execute search for each topic
for rawTopic in rawtopics:
    jsonTopic = rawTopic.toJsonObj()
    topic_id = rawTopic.getNumber()
    age = rawTopic.getAge()
    gender = rawTopic.getGender()
    disease = ','.join(rawTopic.getDiseaseList())
    gene = ','.join(rawTopic.getGeneList)
    other = rawTopic.getOther()

    queryBody = {
        "query" : {
            "filtered" : {
                "filter" : {
                    "range" : {
                        "minimum_age" : { "lte" : age },
                        "maximum_age" : { "gte" : age }
                    }
                },
                "query" : {
                    "match" : {
                        "brief_title" :     [disease, gene, other],
                        "official_title" :  [disease, gene, other],
                        "brief_summary" :   [disease, gene, other]
                        # "operator" : "or" # or | and # https://elasticsearch.cn/book/elasticsearch_definitive_guide_2.x/match-multi-word.html
                        # "minimum_should_match" : "75%"
                    }
                }
            }
        }
    }
    result = es.search(index='clinicaltrials', doc_type='trial', body=queryBody)

    with open(resultDir, 'a') as resultFile:
        resultFile.write('topic_id = ' + str(topic_id))
        resultFile.write(result)
