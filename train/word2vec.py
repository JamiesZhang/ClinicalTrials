#!/usr/bin/env python

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

import os
import smart_open

curDir = os.path.dirname(os.path.abspath(__file__))
docDir = os.path.join(os.path.dirname(curDir), "data/clinicaltrials_txt")

modelDir = os.path.join(os.path.dirname(curDir), "models")
modelFile = "doc2vec.model"

def __getAllDocPath(dirPath, pathList):
    for fname in os.listdir(dirPath):    
        fPath = os.path.join(dirPath, fname)
        if os.path.isdir(fPath):
            __getAllDocPath(fPath, pathList)
        else:
            pathList.append(fPath)

__pathList = []
__getAllDocPath(docDir, __pathList)

def __readCorpus():
    for i, fpath in enumerate(__pathList):
        with smart_open.smart_open(fpath, encoding="utf-8") as f:
            content = ""
            for line in f:
                content = content + line
            yield TaggedDocument(simple_preprocess(content), [i])

def __train():
    # build train corpus
    trainCorpus = list(__readCorpus())

    model = Doc2Vec(vector_size=50, window=2, min_count=2, workers=4, epochs=2)

    # build vocabulary
    model.build_vocab(trainCorpus)

    # train
    model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)

    # save the Doc2Vec Model into model directory
    modelPath = os.path.join(modelDir, modelFile)
    model.save(modelPath)

    return model

def __load():
    modelPath = os.path.join(modelDir, modelFile)
    if os.path.exists(modelPath):
        model = Doc2Vec.load(modelPath)
    else:
        model = __train()
    return model

__model = __load()

def query(queryStr, topn):
    inferredVector = __model.infer_vector(simple_preprocess(queryStr))
    sims = __model.docvecs.most_similar([inferredVector], topn=topn)

    # list of tuple (docid, similarity)
    return sims

def getDocPath(docid):
    return __pathList[docid]

print(len(__model.docvecs))
sims = query("Just a test HIV HIV", 50)
print(sims[0], getDocPath(sims[0][0]))
