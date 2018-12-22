#!/usr/bin/env python

import os

curDir = os.path.dirname(__file__)
dataDir = os.path.join(os.path.dirname(curDir), "data")

__rawDataFile = "qrels-final-trials.txt"
__trainDataFile = "trainset.txt"

__topicsNum = 30
__relevanceLevelNum = 3

def __hasTrainset():
    trainsetPath = os.path.join(dataDir, __trainDataFile)
    return os.path.exists(trainsetPath)

def __loadRawData():
    # rawData list has 30 topic lists
    # each topic list has 3 relevance lists
    # each relevance list has a list of docIDs
    rawData = []
    for i in range(__topicsNum):
        rawData.append([])
        for j in range(__relevanceLevelNum):
            rawData[i].append([])

    # load raw data into rawData list
    rawPath = os.path.join(dataDir, __rawDataFile)
    with open(rawPath, 'r') as fp:
        curLine = fp.readline()
        while curLine:
            # queryID(1~30) 0 docID relevanceLevel(0~2)
            curItems = curLine.split(sep=' ')
            curQueryID = int(curItems[0])
            curDocID = curItems[2]
            curRelevanceLevel = int(curItems[3])

            # append curDocID
            rawData[curQueryID - 1][curRelevanceLevel].append(curDocID)

            # get next line
            curLine = fp.readline()

    return rawData

def __saveTrainset(trainData):
    trainsetPath = os.path.join(dataDir, __trainDataFile)
    with open(trainsetPath, 'w') as fp:
        for i in range(len(trainData)):
            queryID = str(i+1)
            for j in range(len(trainData[i])):
                largerDocID, smallerDocID = trainData[i][j]
                lineStr = ' '.join([queryID,largerDocID,smallerDocID]) + '\n'
                fp.write(lineStr)

def loadTrainset():
    # trainData list has 30 topic lists
    # each topic list has a list of ranking learning positive samples
    # each sample is a tuple with specific format: (larger doc, smaller doc)
    trainData = []
    for i in range(__topicsNum):
        trainData.append([])

    if not __hasTrainset():
        # load rawData into list
        rawData = __loadRawData()

        # generate positive samples for ranking learning
        for i in range(__topicsNum):
            relevanceList0 = rawData[i][0]
            relevanceList1 = rawData[i][1]
            relevanceList2 = rawData[i][2]

            # relevance 1 > relevance 0
            for largerDocID in relevanceList1:
                for smallerDocID in relevanceList0:
                    tempSample = (largerDocID, smallerDocID)
                    trainData[i].append(tempSample)

            # relevance 2 > relevance 1
            for largerDocID in relevanceList2:
                for smallerDocID in relevanceList1:
                    tempSample = (largerDocID, smallerDocID)
                    trainData[i].append(tempSample)

        # save train dataset
        __saveTrainset(trainData)
    else:
        # load already saved train dataset
        trainsetPath = os.path.join(dataDir, __trainDataFile)
        with open(trainsetPath, 'r') as fp:
            curLine = fp.readline()
            while curLine:
                curItems = curLine.split(' ')
                curQueryID = int(curItems[0])
                curLargerDocID = curItems[1]
                curSmallerDocID = curItems[2]

                # append training sample into trainData list
                trainData[curQueryID-1].append((curLargerDocID,curSmallerDocID))

                # get next line
                curLine = fp.readline()

    return trainData