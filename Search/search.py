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

def queryBody(query, boost_brief_title, boost_official_title, boost_brief_summary):
    body = {
        "multi_match" : {
            "query" : query,
            "type" : "best_fields",
            "fields" : ["brief_title"+"^"+str(boost_brief_title), 
                        "official_title"+"^"+str(boost_official_title), 
                        "brief_summary"+"^"+str(boost_brief_summary)],
            # "tie_breaker" : 0.3,
            "minimun_should_match" : "30%"
        }
    }
    return body

def queryOnBrief_title(query):
    body = {
        "query" : {
            "match": {
                "brief_title": query
            }
        }
    }
    return body

def queryOnOfficial_title(query):
    body = {
        "query" : {
            "match": {
                "official_title": query
            }
        }
    }
    return body

def queryOnBrief_summary(query):
    body = {
        "query" : {
            "match": {
                "brief_summary": query
            }
        }
    }
    return body

rawtopics = topics.loadRawTopics()

def getAllTopicScore(): 
    # load query(topics)
    # execute search for each topic
    for rawTopic in rawtopics:
        # jsonTopic = rawTopic.toJsonObj()
        topic_id = rawTopic.getNumber()
        # age = rawTopic.getAge()
        # gender = rawTopic.getGender()
        disease = ','.join(rawTopic.getDiseaseList())
        gene = ','.join(rawTopic.getGeneList())
        other = rawTopic.getOther()
        query = disease + gene + other
        result = es.search(index='clinicaltrials',doc_type='trial', body=queryBody(query,1,1,1), explain=True)
        # print(result)
        # print(result['hits'])
        with open(resultDir, 'a') as resultFile:
            resultFile.write('topic_id = {}\n'.format(str(topic_id)))
            for hit in result['hits']['hits']:
                resultFile.write('id = {}; score = {}\n'.format(str(hit['_id']), str(hit['_score'])))
            resultFile.write('\n')

def getScore(whichIndex, docId, queryBody):
    res = es.explain(index=whichIndex, doc_type='trial', id=docId, body=queryBody)
    score = res['explanation']["value"]
    return score


def subScoreList(queryTopicId, docId, whichIndex):
    sList = []
    disease = ','.join(rawtopics[queryTopicId].getDiseaseList())
    gene = ','.join(rawtopics[queryTopicId].getGeneList())
    other = rawtopics[queryTopicId].getOther()

    # for disease
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(disease)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(disease)))
    sList.append(getScore(whichIndex, docId, queryOnBrief_summary(disease)))

    # for gene
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(gene)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(gene)))
    sList.append(getScore(whichIndex, docId, queryOnBrief_summary(gene)))

    # for other
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(other)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(other)))
    sList.append(getScore(whichIndex, docId, queryOnBrief_summary(other)))

    return sList

# return a list for traning
def getScoreList(queryTopicId, docId):
    id = queryTopicId - 1
    List = subScoreList(id, docId, "clinicaltrials_bm25")
    list2 = subScoreList(id, docId, "clinicaltrials_tfidf")
    List.extend(list2)
    return List
