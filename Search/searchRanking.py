#!/usr/bin/env python

# use boost(get from train module) to search and get a rank of documents
import sys
import os
curDir = os.path.dirname(__file__)
parentDir = os.path.dirname(curDir)
sys.path.append(parentDir)
from preprocess import topics,docs, rankingDataset
from elasticsearch import Elasticsearch
import requests
from train import mp

curDir = os.path.dirname(os.path.abspath(__file__)) #this way, right
parentDir = os.path.dirname(curDir)
dataDir = os.path.join(parentDir, 'data')

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

# for a query topicId, get all result :{docID：sorce}
def getResultList(topicId, method, topicBoostList, docBoostList, methodBoost):
    resDic = {}
    Results = mySearch(method, topicId, topicBoostList, docBoostList)
    for hit in Results:
        res = {hit["_id"] : hit["_score"]*methodBoost}
        resDic.update(res)
    return resDic

def resultToFile(moduleId, topicList, methodBoostList, topicBoostList, docBoostList):
    bm25Boost = methodBoostList[0]
    tfidfBoost = methodBoostList[1]

    bm25TopicBoostList = topicBoostList[:3]
    tfidfTopicBoostList = topicBoostList[3:]

    bm25DocBoostList = docBoostList[:9]
    tfidfDocBoostList = docBoostList[9:]

    bm25Result = {}
    tfidfResult = {}
    finalResult = {}
    for topicId in topicList:
        topicId -= 1
        bm25Result =  getResultList(topicId, bm25Index, bm25TopicBoostList, bm25DocBoostList, bm25Boost)
        tfidfResult = getResultList(topicId, tfidfIndex, tfidfTopicBoostList, tfidfDocBoostList, tfidfBoost)
        for docId in tfidfResult.keys():
            if docId in bm25Result.keys():
                finalScore = bm25Result[docId] + tfidfResult[docId]
                bm25Result.update({docId : finalScore})
            else:
                finalScore = tfidfResult[docId]
                bm25Result.update({docId : finalScore})
        finalResult = bm25Result
        finalResult= sorted(finalResult.items(), key=lambda d:d[1], reverse = True)   # sort by score
        with open(os.path.join(dataDir, 'res{}.txt'.format(moduleId)),'a') as f:
            r = 0
            for res in finalResult:
                f.write(' '.join([str(topicId), "Q0", res[0], str(r), str(res[1]), "SZIR"]))
                r += 1

for module in range(5):
    docBoostList, topicBoostList, methodBoostList = mp.getWeights(module)
    topicList = rankingDataset.getTopicIDsForTest(module)
    resultToFile(module, topicList, methodBoostList, topicBoostList, docBoostList)
    print('finish module {}'.format(module))