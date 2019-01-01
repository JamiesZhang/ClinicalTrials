#!/usr/bin/env python

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.matutils import unitvec
from numpy import dot
from preprocess.topics import getTopicByID

import os
import smart_open

curDir = os.path.dirname(os.path.abspath(__file__))
docDir = os.path.join(os.path.dirname(curDir), "data/clinicaltrials_txt")

modelDir = os.path.join(os.path.dirname(curDir), "models")
__modelFile = "doc2vec.model"

def __getAllDocPath(dirPath, pathList):
    for fname in os.listdir(dirPath):    
        fPath = os.path.join(dirPath, fname)
        if os.path.isdir(fPath):
            __getAllDocPath(fPath, pathList)
        else:
            pathList.append(fPath)

__pathList = []
__getAllDocPath(docDir, __pathList)

__docIDList = []
for path in __pathList:
    docID = path.split(os.sep)[-1].split('.')[0]
    __docIDList.append(docID)

__docIndexDict = {}
for index in range(len(__docIDList)):
    __docIndexDict[__docIDList[index]] = index

def __readCorpus():
    for index, fpath in enumerate(__pathList):
        with smart_open.smart_open(fpath, encoding="utf-8") as f:
            content = ""
            for line in f:
                content = content + line
            yield TaggedDocument(simple_preprocess(content), [index])

def __train():
    # build train corpus
    trainCorpus = list(__readCorpus())

    model = Doc2Vec(vector_size=50, window=2, min_count=2, workers=4, epochs=2)

    # build vocabulary
    model.build_vocab(trainCorpus)

    # train
    model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)

    # save the Doc2Vec Model into model directory
    modelPath = os.path.join(modelDir, __modelFile)
    model.save(modelPath)

    return model

def __load():
    modelPath = os.path.join(modelDir, __modelFile)
    if os.path.exists(modelPath):
        print("Load the model of word2vec...")
        model = Doc2Vec.load(modelPath)
        print("Success!")
    else:
        print("Train the model of word2vec...")
        model = __train()
        print("Success!")
    return model

__model = __load()

def query(queryStr, topn):
    inferredVector = __model.infer_vector(simple_preprocess(queryStr))
    sims = __model.docvecs.most_similar([inferredVector], topn=topn)

    # list of tuple (docid, similarity)
    return sims

def similarity(topicID, docID):
    curTopic = getTopicByID(topicID)
    queryStr = curTopic.toQueryStr()

    docIndex = getDocIndexes([docID])[0]
    docVector = __model.docvecs[docIndex]
    inferredVector = __model.infer_vector(simple_preprocess(queryStr))

    return dot(unitvec(docVector), unitvec(inferredVector))

def getDocIDs(docIndexes):
    docIDs = []
    for docIndex in docIndexes:
        docIDs.append(__docIDList[docIndex])
    return docIDs

def getDocIndexes(docIDs):
    docIndexes = []
    for docID in docIDs:
        docIndexes.append(__docIndexDict.get(docID))
    return docIndexes

# print(len(__model.docvecs))
# sims = query("Just a test HIV HIV", 50)
# print(sims[0], getDocPath(sims[0][0]))
