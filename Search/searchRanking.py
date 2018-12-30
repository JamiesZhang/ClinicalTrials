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
from train import mp, word2vec
import math, numpy

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

def baseBody(queryTopicId):
    disease = ','.join(rawtopics[queryTopicId].getDiseaseList())
    gene = ','.join(rawtopics[queryTopicId].getGeneList())
    other = rawtopics[queryTopicId].getOther()
    bBody = {
            "query" : {
                "bool" : {
                    "must" : [
                        {
                            "multi_match" : {
                                "query" : disease,
                                "fields" : ["brief_title^2", 
                                            "official_title", 
                                            "textblock", "mesh_term", "condition", "keyword"],
                                "tie_breaker" : 0.3,
                                "boost" : 1.8
                            }
                        },
                        {
                            "multi_match" : {
                            "query" : gene,
                            "fields" : ["brief_title^2", 
                                            "official_title", 
                                            "textblock", "mesh_term", "condition", "keyword"],
                            "tie_breaker" : 0.3,
                            # "boost" : 1.5
                            }
                        },
                        {
                                "multi_match" : {
                                "query" : other,
                                "fields" : ["brief_title^2", 
                                            "official_title", 
                                            "textblock", "mesh_term", "condition", "keyword"],
                                "tie_breaker" : 0.3,
                                # "boost" : 1
                            }
                        }
                    ],
                    "should" : [
                        {
                            "term" : {
                                "textblock" : disease
                            }
                        },
                        {
                             "term" : {
                                "keyword" : disease
                            }
                        }
                    ]
                }
            }
        }
    return bBody

def queryBody(queryTopicId, topicBoostList, docBoostList):
    disease = ','.join(rawtopics[queryTopicId].getDiseaseList())
    gene = ','.join(rawtopics[queryTopicId].getGeneList())
    other = rawtopics[queryTopicId].getOther()
    body = {
        "query" : {
            "bool" : {
                "should" : [
                    {
                        "multi_match" : {
                            "query" : disease,
                            "fields" : ["brief_title"+"^"+str(docBoostList[0]), 
                                        "official_title"+"^"+str(docBoostList[1]), 
                                        "textblock"+"^"+str(docBoostList[2]),
                                        "mesh_term"+"^"+str(docBoostList[3]),
                                        "condition"+"^"+str(docBoostList[4]),
                                        "keyword"+"^"+str(docBoostList[5])],
                            # "tie_breaker" : 0.3,
                            "boost" : topicBoostList[0]
                        }
                    },
                    {
                            "multi_match" : {
                            "query" : gene,
                            "fields" : ["brief_title"+"^"+str(docBoostList[6]), 
                                        "official_title"+"^"+str(docBoostList[7]), 
                                        "textblock"+"^"+str(docBoostList[8]),
                                        "mesh_term"+"^"+str(docBoostList[9]),
                                        "condition"+"^"+str(docBoostList[10]),
                                        "keyword"+"^"+str(docBoostList[11])],
                            # "tie_breaker" : 0.3,
                            "boost" : topicBoostList[1]
                        }
                    },
                    {
                            "multi_match" : {
                            "query" : other,
                            "fields" : ["brief_title"+"^"+str(docBoostList[12]), 
                                        "official_title"+"^"+str(docBoostList[13]), 
                                        "textblock"+"^"+str(docBoostList[14]),
                                        "mesh_term"+"^"+str(docBoostList[15]),
                                        "condition"+"^"+str(docBoostList[16]),
                                        "keyword"+"^"+str(docBoostList[17])],
                            # "tie_breaker" : 0.3,
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
    result = es.search(index = index, doc_type='trial', body=baseBody(topicId), size=500)['hits']['hits']
    return result

# for a query topicId, get all result :{docID：sorce}
def getResultList(topicId, method, topicBoostList, docBoostList, methodBoost):
    resDic = {}
    Results = mySearch(method, topicId, topicBoostList, docBoostList)
    for hit in Results:
        res = {hit["_id"] : hit["_score"]*methodBoost}
        resDic.update(res)
    return resDic

def getBaseResultList(topicId, whichIndex):
    resDic = []
    Results = es.search(index = whichIndex, doc_type='trial', body=baseBody(topicId), size=500)['hits']['hits']
    for hit in Results:
        res = [hit["_id"] , hit["_score"]]
        resDic.append(res)
    return resDic

def relu(num):
    n = 0
    if num < 0:
        n = 0
    else:
        n = num
    return n

def sigmoid(num):
    return 1/(1+math.pow(math.e, -num))

def resultToFile(moduleId, topicList, methodBoostList, topicBoostList, docBoostList, t):
    bm25Boost = methodBoostList[0]
    tfidfBoost = methodBoostList[1]

    bm25TopicBoostList = topicBoostList[:3]
    tfidfTopicBoostList = topicBoostList[3:]

    bm25DocBoostList = docBoostList[:18]
    tfidfDocBoostList = docBoostList[18:]

    bm25Result = {}
    tfidfResult = {}

    returnResult = []
    f = open(os.path.join(dataDir, 'res{}.txt'.format(moduleId)),'w')
    for topicId in topicList:
        finalResult = {}
        topicID = topicId
        topicID -= 1
        bm25Result =  getResultList(topicID, bm25Index, bm25TopicBoostList, bm25DocBoostList, bm25Boost)
        tfidfResult = getResultList(topicID, tfidfIndex, tfidfTopicBoostList, tfidfDocBoostList, tfidfBoost)
        for docId in tfidfResult.keys():
            if docId in bm25Result.keys():
                finalScore = bm25Result[docId] + tfidfResult[docId]
            else:
                finalScore = tfidfResult[docId]
            s = word2vec.similarity(topicID, docId)
            finalScore = sigmoid(finalScore)*math.log(3 + relu(t*s))
            bm25Result.update({docId : finalScore})
        finalResult = bm25Result
        finalResult= sorted(finalResult.items(), key=lambda d:d[1], reverse = True)   # sort by score
        r = 0  # ranking number
        for res in finalResult:
            f.write(' '.join([str(topicID+1), "Q0", res[0], str(r), str(res[1]), "SZIR"]) + '\n')
            returnResult.append([topicID+1, {res[0]:res[1]}])
            r += 1
    f.close()
    return returnResult

def baseResultToFile(moduleId, topicList):
    returnResult = []
    bm25_result = []
    tfidf_result = []
    bm25File = open(os.path.join(dataDir, 'baseResBM25{}.txt'.format(moduleId)),'w')
    tfidfFile = open(os.path.join(dataDir, 'baseResTfidf{}.txt'.format(moduleId)),'w')
    baseFile = open(os.path.join(dataDir, 'baseRes{}.txt'.format(moduleId)),'w')
    for topicId in topicList:
        topicID = topicId
        topicID -= 1
        bm25_result =  getBaseResultList(topicID, bm25Index)
        tfidf_result = getBaseResultList(topicID, tfidfIndex)
        # bm25Result = {}
        # tfidfResult = {}

        # for r in bm25_result:
        #     bm25Result.update({r[0]:r[1]})

        # for rr in tfidf_result:
        #     tfidfResult.update({rr[0]:rr[1]})

        # for docId in tfidfResult.keys():
        #     if docId in bm25Result.keys():
        #         finalScore = bm25Result[docId] + tfidfResult[docId]
        #     else:
        #         finalScore = tfidfResult[docId]
        #     # s = word2vec.similarity(topicID, docId)
        #     # finalScore = finalScore*math.log(math.e + relu(t*s))
        #     bm25Result.update({docId : finalScore})
        # finalResult = bm25Result
        # finalResult= sorted(finalResult.items(), key=lambda d:d[1], reverse = True)   # sort by score
        # r = 0  # ranking number
        # for res in finalResult:
        #     baseFile.write(' '.join([str(topicID+1), "Q0", res[0], str(r), str(res[1]), "SZIR"]) + '\n')
        #     returnResult.append([topicID+1, {res[0]:res[1]}])
        #     r += 1

        r1 = 0
        for res in bm25_result:
            bm25File.write(' '.join([str(topicID+1), "Q0", res[0], str(r1), str(res[1]), "SZIR"]) + '\n')
            r1 += 1
        
        r2 = 0
        for res in tfidf_result:
            tfidfFile.write(' '.join([str(topicID+1), "Q0", res[0], str(r2), str(res[1]), "SZIR"]) + '\n')
            r2 += 1
    bm25File.close()
    tfidfFile.close()
    baseFile.close()
    return returnResult
    
#def getFinalResult(moduleId, res1, res2, p):

#    fRes = {}
#    finalFile = open(os.path.join(dataDir, 'FinalScore{}.txt'.format(moduleId)),'w')
#    for i in range(30):
#        j = i+1
#        for r1, r2 in res1, res2:
#            if r1[0] == j and r2[0] == j:
#                for docId in r2[1].keys():
#                    if docId in r1[1].keys():
#                        finalScore = p*r1[1][docId] + (1-p)*r2[1][docId]
#                    else:
#                        finalScore = r2[1][docId]
#                    fRes.update({docId : finalScore})
#                    finalResult = fRes
#                    finalResult= sorted(finalResult.items(), key=lambda d:d[1], reverse = True)   # sort by score
#                    x = 0  # ranking number
#                    for res in finalResult:
#                        finalFile.write(' '.join([str(j), "Q0", res[0], str(x), str(res[1]), "SZIR"]) + '\n')
#                        x += 1
#            else:
#                break
#    finalFile.close()

for module in range(5):
    weight = mp.getWeights(module)
    docBoostList = weight[0]
    topicBoostList = weight[1]
    methodBoostList = weight[2]
    t = weight[3]
    topicList = rankingDataset.getTopicIDsForTest(module)
    resT = resultToFile(module, topicList, methodBoostList, topicBoostList, docBoostList, t)
    rBase = baseResultToFile(module,topicList)
    p = 0.5        # 一个需要手调的参数，衡量
    #getFinalResult(module, resT, rBase, p)
    print('finish module {}'.format(module))

