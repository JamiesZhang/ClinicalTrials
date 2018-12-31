#!/usr/bin/env python

from preprocess import docs, topics, rankingDataset
from train import word2vec, mp
from Search import bulkCreatIndex, search
#from TermExt import termExtension

from Search.searchRanking import *

for module in range(5):
    weight = mp.getWeights(module)
    docBoostList = weight[0]
    topicBoostList = weight[1]
    methodBoostList = weight[2]
    t = weight[3]
    topicList = rankingDataset.getTopicIDsForTest(module)
    resT = resultToFile(module, topicList, methodBoostList, topicBoostList, docBoostList, t)
    rBase = baseResultToFile(module,topicList)
    p = 0.6        
    getFinalResult(module, topicList, resT, rBase, p)
    print('finish module {}'.format(module))
