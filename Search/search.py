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

bm25Index = "clinicaltrials_bm25"
tfidfIndex = "clinicaltrials_tfidf"

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

# def queryOnBrief_summary(query):
#     body = {
#         "query" : {
#             "match": {
#                 "brief_summary": query
#             }
#         }
#     }
#     return body

def queryOnTextblock(query):
    body = {
        "query" : {
            "match": {
                "textblock": query
            }
        }
    }
    return body

def queryOnMesh_term(query):
    body = {
        "query" : {
            "match": {
                "mesh_term": query
            }
        }
    }
    return body

def queryOnCondition(query):
    body = {
        "query" : {
            "match": {
                "condition": query
            }
        }
    }
    return body

def queryOnKeyword(query):
    body = {
        "query" : {
            "match": {
                "keyword": query
            }
        }
    }
    return body

def queryOnXXX(query, XXX):
    body = {
        "query" : {
            "match": {
                XXX: query
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
        result = es.search(index='clinicaltrials_bm25',doc_type='trial', body=queryBody(query,1,1,1), explain=True)
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

def getResScore(whichIndex, queryBody):
    resDict = {}
    Results = es.search(index=whichIndex, doc_type='trial',body=queryBody)["hits"]["hits"]
    for hit in Results:
        res = {hit["_id"] : hit["_score"]}
        resDict.update(res)
    return resDict

def subScoreList(queryTopicId, docId, whichIndex):
    sList = []
    disease = ','.join(rawtopics[queryTopicId].getDiseaseList())
    gene = ','.join(rawtopics[queryTopicId].getGeneList())
    other = rawtopics[queryTopicId].getOther()

    # for disease
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(disease)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(disease)))
    sList.append(getScore(whichIndex, docId, queryOnTextblock(disease)))
    sList.append(getScore(whichIndex, docId, queryOnMesh_term(disease)))
    sList.append(getScore(whichIndex, docId, queryOnCondition(disease)))
    sList.append(getScore(whichIndex, docId, queryOnKeyword(disease)))

    # for gene
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(gene)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(gene)))
    sList.append(getScore(whichIndex, docId, queryOnTextblock(gene)))
    sList.append(getScore(whichIndex, docId, queryOnMesh_term(gene)))
    sList.append(getScore(whichIndex, docId, queryOnCondition(gene)))
    sList.append(getScore(whichIndex, docId, queryOnKeyword(gene)))

    # for other
    sList.append(getScore(whichIndex, docId, queryOnBrief_title(other)))
    sList.append(getScore(whichIndex, docId, queryOnOfficial_title(other)))
    sList.append(getScore(whichIndex, docId, queryOnTextblock(other)))
    sList.append(getScore(whichIndex, docId, queryOnMesh_term(other)))
    sList.append(getScore(whichIndex, docId, queryOnCondition(other)))
    sList.append(getScore(whichIndex, docId, queryOnKeyword(other)))

    return sList

# return a list for traning
def getScoreList(queryTopicId, docId):
    id = queryTopicId - 1
    List = subScoreList(id, docId, "clinicaltrials_bm25")
    list2 = subScoreList(id, docId, "clinicaltrials_tfidf")
    List.extend(list2)
    return List

def getSubScoreDict(module, topicId, docId):
    sList = []
    disease = ','.join(rawtopics[topicId].getDiseaseList())
    gene = ','.join(rawtopics[topicId].getGeneList())
    other = rawtopics[topicId].getOther()
    
    docFieldList = ["brief_title", "official_title", "textblock", "mesh_term", "condition", "keyword"]
    topicFieldList = [disease, gene, other]

    resDict = []

    for i in range(18):
        if i < 6:
            n = 0
        elif i < 12:
            n = 1
        else:
            n = 2
        resDict.append(getResScore(whichIndex=module, queryBody=queryOnXXX(topicFieldList[n],docFieldList[i % 6])))


    for i in range(len(resDict)):
        if docId in resDict[i].keys():
            sList.append(resDict[i][docId])
        else:
            if i < 6:
                tNumber = 0
            elif i < 12:
                tNumber = 1
            else:
                tNumber = 2
            s = getScore(whichIndex=module, docId=docId, queryBody=queryOnXXX(topicFieldList[tNumber],docFieldList[i % 6]))
            sList.append(s)
            
    return sList

def getScoreDict(topicId, docIdList):
    resDict = {}
    for docId in docIdList:
        list1 = getSubScoreDict(bm25Index, topicId, docId)
        list2 = getSubScoreDict(tfidfIndex, topicId, docId)
        list1.extend(list2)
        resDict.update({docId:list1})
    return resDict




# def test(topicid, docID):
#     disease = ','.join(rawtopics[topicid].getDiseaseList())
#     # print(getRes("clinicaltrials_bm25", queryOnTextblock(disease))["_score"])
#     print(getScore("clinicaltrials_bm25", docID, queryOnTextblock(disease)))
# # 'NCT00002575', '_score': 9.86981
# # 'NCT00003648', '_score': 9.649702,
# test(1, "NCT00003648")

    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnBrief_title(disease)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(disease)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(disease)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(disease)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(disease)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnCondition(disease)))

    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnBrief_title(gene)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(gene)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(gene)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(gene)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(gene)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnCondition(gene)))

    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnBrief_title(other)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(other)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(other)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnMesh_term(other)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnOfficial_title(other)))
    # resDict.append(getResScore(whichIndex=module, queryBody=queryOnCondition(other)))
    