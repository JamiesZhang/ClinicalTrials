#!/usr/bin/env python

import os
import math

curDir = os.path.dirname(__file__)
dataDir = os.path.join(os.path.dirname(curDir), "data")

__rawDataFile = "qrels-final-trials.txt"
__rankingDataFilePrefix = "rankingDataset"
__qrelsFilePrefix = "qrels"

__topicsNum = 30
__relevanceLevelNum = 3

__foldNum = 5
__topicsNumPerFold = 6
__topicFolds = [[28, 29, 25, 22, 6, 7],
                [26, 11, 1, 18, 21, 4],
                [19, 24, 27, 30, 12, 23],
                [13, 14, 3, 16, 8, 9],
                [15, 20, 5, 10, 17, 2]]

# rankingData list has 5 fold lists
# each fold list 6 topic lists
# each topic list has a list of ranking learning positive samples
# each sample is a tuple with specific format: (larger doc, smaller doc)
__rankingData = []
for i in range(__foldNum):
    __rankingData.append([])
    for j in range(__topicsNumPerFold):
        __rankingData[i].append([])

def __hasDataset():
    for foldID in range(__foldNum):
        curDataFile = "{}{}.txt".format(__rankingDataFilePrefix, foldID)
        curDataPath = os.path.join(dataDir, curDataFile)
        if not os.path.exists(curDataPath):
            return False
    return True

def __loadRawDataAndSaveQrels():
    # rawData list has 30 topic lists
    # each topic list has 3 relevance lists
    # each relevance list has a list of docIDs
    rawData = []
    for i in range(__topicsNum):
        rawData.append([])
        for j in range(__relevanceLevelNum):
            rawData[i].append([])

    # prepare for saving qrels
    fps = [None]*5
    for i in range(__foldNum):
        curQrelsFile = "{}{}.txt".format(__qrelsFilePrefix, i)
        curQrelsPath = os.path.join(dataDir, curQrelsFile)
        fps[i] = open(curQrelsPath, 'w')

    # load raw data into rawData list
    rawPath = os.path.join(dataDir, __rawDataFile)
    with open(rawPath, 'r') as fp:
        curLine = fp.readline()
        while curLine:
            # queryID(1~30) 0 docID relevanceLevel(0~2)
            curItems = curLine.split(sep=' ')
            curTopicID = int(curItems[0])
            curDocID = curItems[2]
            curRelevanceLevel = int(curItems[3])

            # append curDocID
            rawData[curTopicID - 1][curRelevanceLevel].append(curDocID)

            # save qrels
            curFoldID, _ = __getFoldAndIndexByTopic(curTopicID)
            fps[curFoldID].write(curLine)

            # get next line
            curLine = fp.readline()

    # close fps for qrels
    for i in range(__foldNum):
        fps[i].close()

    return rawData

def __saveDataset():
    for foldID in range(__foldNum):
        curDataFile = "{}{}.txt".format(__rankingDataFilePrefix, foldID)
        curDataPath = os.path.join(dataDir, curDataFile)
        with open(curDataPath, 'w') as fp:
            for indexID in range(__topicsNumPerFold):
                curTopicID = __topicFolds[foldID][indexID]
                curSampleList = __rankingData[foldID][indexID]
                for i in range(len(curSampleList)):
                    largerDocID, smallerDocID = curSampleList[i]
                    lineStr = ' '.join([str(curTopicID),largerDocID,smallerDocID]) + '\n'
                    fp.write(lineStr)

def __getIndexByTopic(foldID, topicID):
    curIndexID = -1
    for indexID in range(__topicsNumPerFold):
        if __topicFolds[foldID][indexID] == topicID:
            curIndexID = indexID
            break
    return curIndexID

def __getFoldAndIndexByTopic(topicID):
    curFoldID = -1
    curIndexID = -1
    for foldID in range(__foldNum):
        indexID = __getIndexByTopic(foldID, topicID)
        if indexID != -1:
            curFoldID = foldID
            curIndexID = indexID
            break
    return curFoldID, curIndexID

def __loadDataset():
    if not __hasDataset():
        # load rawData into list
        rawData = __loadRawDataAndSaveQrels()

        # generate positive samples for ranking learning
        for i in range(__topicsNum):
            relevanceList0 = rawData[i][0]
            relevanceList1 = rawData[i][1]
            relevanceList2 = rawData[i][2]

            # get corresponding fold ID (0~4) and index ID (0~5)
            curTopicID = i+1
            curFoldID, curIndexID = __getFoldAndIndexByTopic(curTopicID)
            if curFoldID == -1:
                raise RuntimeError("Invalid fold id while processing rawData in rankingDataset.py!")

            curSampleList = __rankingData[curFoldID][curIndexID]

            # relevance 1 > relevance 0
            for largerDocID in relevanceList1:
                for smallerDocID in relevanceList0:
                    tempSample = (largerDocID, smallerDocID)
                    curSampleList.append(tempSample)

            # relevance 2 > relevance 1
            for largerDocID in relevanceList2:
                for smallerDocID in relevanceList1:
                    tempSample = (largerDocID, smallerDocID)
                    curSampleList.append(tempSample)

        # save data dataset
        __saveDataset()
    else:
        # load already saved ranking dataset
        for foldID in range(__foldNum):
            curDataFile = "{}{}.txt".format(__rankingDataFilePrefix, foldID)
            curDataPath = os.path.join(dataDir, curDataFile)
            with open(curDataPath, 'r') as fp:
                curLine = fp.readline()
                while curLine:
                    curItems = curLine.split(' ')
                    curTopicID = int(curItems[0])
                    curLargerDocID = curItems[1]
                    curSmallerDocID = curItems[2]

                    # get corresponding fold ID (0~4) and index ID (0~5)
                    curIndexID = __getIndexByTopic(foldID, curTopicID)
                    if curIndexID == -1:
                        raise RuntimeError("Invalid index ID while loading rankingData in rankingDataset.py!")

                    # append training sample into rankingData list
                    __rankingData[foldID][curIndexID].append((curLargerDocID,curSmallerDocID))

                    # get next line
                    curLine = fp.readline()

# modelID is in the range of 0~4
# NOTE: dont't change the return list (readonly) otherwise you'll modify the rankingData!!!
# NOTE: dont't change the return list (readonly) otherwise you'll modify the rankingData!!!
# NOTE: dont't change the return list (readonly) otherwise you'll modify the rankingData!!!
def constructDatasetForModel(modelID):
    if modelID < 0 or modelID >= __foldNum:
        raise RuntimeError("Invalid modelID {}!".format(modelID))

    validationID = modelID % __foldNum # 1 fold
    testID = (modelID+1) % __foldNum # 1 fold

    trainIDs = [(modelID+2) % __foldNum, (modelID+3) % __foldNum, (modelID+4) % __foldNum] # 3 folds

    validationDataset = __rankingData[validationID] # 6 elements
    testDataset = __rankingData[testID] # 6 elements

    trainDataset = [] # 18 elements
    for trainID in trainIDs:
        for indexID in range(__topicsNumPerFold):
            trainDataset.append(__rankingData[trainID][indexID])

    return validationDataset, testDataset, trainDataset

def getTopicIDForValidation(modelID, indexID):
    if modelID < 0 or modelID >= __foldNum:
        raise RuntimeError("Invalid modelID {}!".format(modelID))
    if indexID < 0 or indexID >= __topicsNumPerFold:
        raise RuntimeError("Invalid indexID {}!".format(indexID))

    foldID = modelID % __foldNum
    return __topicFolds[foldID][indexID]

def getTopicIDForTest(modelID, indexID):
    if modelID < 0 or modelID >= __foldNum:
        raise RuntimeError("Invalid modelID {}!".format(modelID))
    if indexID < 0 or indexID >= __topicsNumPerFold:
        raise RuntimeError("Invalid indexID {}!".format(indexID))

    foldID = (modelID + 1) % __foldNum
    return __topicFolds[foldID][indexID]

def getTopicIDForTrain(modelID, indexID):
    if modelID < 0 or modelID >= __foldNum:
        raise RuntimeError("Invalid modelID {}!".format(modelID))
    if indexID < 0 or indexID >= __topicsNumPerFold*3:
        raise RuntimeError("Invalid indexID {}!".format(indexID))

    foldID = (modelID + 2 + math.floor(indexID/__topicsNumPerFold)) % __foldNum
    indexID = indexID % __topicsNumPerFold
    return __topicFolds[foldID][indexID]

# Load ranking dataset to initialize rankingData
# If it's the first time to run, we'll load raw data and convert into 5-fold ranking data and save them
# Otherwise we just load already saved ranking data folds
__loadDataset()

