#!/usr/bin/env python

# use boost(get from train module) to search and get a rank of documents

import sys
import os
curDir = os.path.dirname(__file__)
parentDir = os.path.dirname(curDir)
sys.path.append(parentDir)
from preprocess import topics,docs
from elasticsearch import Elasticsearch
import requests

# connect to es ser ver
es = Elasticsearch([{'host':'localhost', 'port' : 9200}])
# test connection
if requests.get(r'http://localhost:9200').status_code != 200:
    raise RuntimeError('connection failure')

bm25Index = "clinicaltrials_bm25"
tfidfIndex = "clinicaltrials_tfidf"
rawtopics = topics.loadRawTopics()

def queryBody(queryTopicId, topicBoostList, docBoostList):
    disease = ','.join(rawtopics[queryTopicId].getDiseaseList())
    gene = ','.join(rawtopics[queryTopicId].getGeneList())
    other = rawtopics[queryTopicId].getOther()
    body = {
        "query" : {
            "bool" : {
                "must" : [
                    {
                        "multi_match" : {
                            "query" : disease,
                            "fields" : ["brief_title"+"^"+str(docBoostList[0]), 
                                        "official_title"+"^"+str(docBoostList[1]), 
                                        "brief_summary"+"^"+str(docBoostList[2])],
                            "tie_breaker" : 0.3,
                            "boost" : topicBoostList[0]
                        }
                    },
                    {
                            "multi_match" : {
                            "query" : gene,
                            "fields" : ["brief_title"+"^"+str(docBoostList[3]), 
                                        "official_title"+"^"+str(docBoostList[4]), 
                                        "brief_summary"+"^"+str(docBoostList[5])],
                            "tie_breaker" : 0.3,
                            "boost" : topicBoostList[1]
                        }
                    },
                    {
                            "multi_match" : {
                            "query" : other,
                            "fields" : ["brief_title"+"^"+str(docBoostList[6]), 
                                        "official_title"+"^"+str(docBoostList[7]), 
                                        "brief_summary"+"^"+str(docBoostList[8])],
                            "tie_breaker" : 0.3,
                            "boost" : topicBoostList[2]
                        }
                    }
                ]
            }
        }
    }
    return body

# return a dict, can get id in "_id", and get score in "_score"
def mySearch(index, topicId, topicBoostList, docBoostList):
    result = es.search(index = index, doc_type='trial', body=queryBody(topicId, topicBoostList, docBoostList), size=50)['hits']['hits']
    return result

# if I got these attribute
methodBoostList = []
topicBoostList = []
docBoostList = []

bm25TopicBoostList = topicBoostList[:3]
tfidfTopicBoostList = topicBoostList[3:]

bm25DocBoostList = docBoostList[:9]
tfidfDocBoostList = docBoostList[9:]

bm25Boost = methodBoostList[0]
tfidfBoost = methodBoostList[1]

bm25ResultList = []
tfidfResultList = []

def getResultList(method, topicBoostList, docBoostList):
    List = []
    for topicId in range(30):
        bm25Results = mySearch(method, id, topicBoostList, docBoostList)
        for hit in bm25Results:
            resList = [topicId , {hit["_id"], hit["_score"]}]
            List.append(resList)
    return List

bm25ResultList = getResultList(bm25Index, bm25TopicBoostList, bm25DocBoostList)
tfidfResultList = getResultList(tfidfIndex, tfidfTopicBoostList, tfidfDocBoostList)

# the final ranking list!!!
finalResultList = []

for topicId in range(30):                           #针对每一个topic查询
    for bm25Result in bm25ResultList:               #每一个查询中的docID和分数
        if bm25Result[0] == topicId:                #如果是当前查询的结果
            for docId in bm25Result[1].keys():      #对于每一个文档doc
                for tfidfResult in tfidfResultList: #寻找tfidf结果中响应的文档
                    if (tfidfResult[0] == topicId) and (docId in tfidfResult[1].keys()): #如果找到了相应的文档，则计算final的分数
                        finalScore = bm25Boost * bm25Result[1][docId] + \
                                     tfidfBoost * tfidfResult[1][docId]
                        finalResult = [topicId, docId, finalScore]
                        finalResultList.append(finalResult)
                    else:                           # 如果没有找到相应的文档，则找下一个文档
                        continue                    # 是用continue还是break？？？？？？？？脑子不转了。。。
        else:       #说明针对于topicID的这个查询查完了，继续下一个TopicID的查询
            break