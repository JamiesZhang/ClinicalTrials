#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
from preprocess import rankingDataset
import time
import math

curDir = os.path.dirname(os.path.abspath(__file__))

modelDir = os.path.join(os.path.dirname(curDir), "models")
__subDirPrefix = "MP"
__modelFilePrefix = "MP"

__searchModelNum = 2
__topicFieldNum = 3
__docFieldNum = 6

__inputTensorXName = "inputx"
__inputTensorYName = "inputy"
__inputTensorWName = "inputw"
__docVariableTensorName = "docVariable"
__topicVariableTensorName = "topicVariable"
__searchVariableTensorName = "searchVariable"
__word2vecVariableTensorName = "word2vecVariable"
__outputTensorName = "output"
__finalTensorName = "train"
__accuracyTensorName = "tensor"

__batchSize = 32
__epochNum = 1

graph = tf.Graph()
sess = tf.Session(graph=graph)

#docVariableInitValue = np.random.randn(36)
#topicVariableInitValue = np.random.randn(6)
#searchVariableInitValue = np.random.randn(2)
#similarityVariableInitValue = np.random.randn(1)

docVariableInitValue = np.array(rankingDataset.priorDocVar, dtype = np.float64)
topicVariableInitValue = np.array(rankingDataset.priorTopicVar, dtype=np.float64)
searchVariableInitValue = np.array(rankingDataset.priorSearchVar, dtype=np.float64)
similarityVariableInitValue = np.array(rankingDataset.priorSimVar, dtype=np.float64)

def __buildModel(modelID):
    
    with graph.as_default():
        # compute dimensions
        docLayerDim = __searchModelNum * __topicFieldNum * __docFieldNum # 36
        topicLayerDim = __searchModelNum * __topicFieldNum # 6
        searchLayerDim = __searchModelNum # 2

        # layer 0: input layer
        inputTensorX = tf.placeholder(shape=(__batchSize, 2, docLayerDim), name=__inputTensorXName+str(modelID), dtype=tf.float64) # n*2*36
        inputTensorY = tf.placeholder(shape=(__batchSize, 1), name=__inputTensorYName+str(modelID), dtype=tf.float64) # n*1
        inputTensorS = tf.placeholder(shape=(__batchSize, 2), name=__inputTensorWName+str(modelID), dtype=tf.float64) # n*2

        # layer 1: doc-field layer
        docVariableTensor = tf.Variable(docVariableInitValue, name=__docVariableTensorName+str(modelID)) # 36
        dotTensor1 = inputTensorX * docVariableTensor # n*2*18
        reshapeTensor1 = tf.reshape(dotTensor1, shape=(-1, 2, topicLayerDim, __docFieldNum)) # n*2*6*3
        reduceSumTensor1 = tf.reduce_sum(reshapeTensor1, 3) # n*2*6
        # reluTensor1 = tf.nn.relu(reduceSumTensor1) # n*2*6

        # layer 2: topic-field layer
        topicVariableTensor = tf.Variable(topicVariableInitValue, name=__topicVariableTensorName+str(modelID)) # 6
        dotTensor2 = reduceSumTensor1 * topicVariableTensor # n*2*6
        reshapeTensor2 = tf.reshape(dotTensor2, shape=(-1, 2, searchLayerDim, __topicFieldNum))  # n*2*2*3
        reduceSumTensor2 = tf.reduce_sum(reshapeTensor2, 3)  # n*2*2
        # reluTensor2 = tf.nn.relu(reduceSumTensor2) # n*2*2

        # layer 3: search-field layer
        searchVariableTensor = tf.Variable(searchVariableInitValue, name=__searchVariableTensorName+str(modelID)) # 2
        dotTensor3 = reduceSumTensor2 * searchVariableTensor  # n*2*2
        reshapeTensor3 = tf.reshape(dotTensor3, shape=(-1, 2, 1, __searchModelNum))  # n*2*1*2
        reduceSumTensor3 = tf.nn.sigmoid(tf.reduce_sum(tf.reduce_sum(reshapeTensor3, 3), 2)) # n*2
        # reluTensor3 = tf.nn.relu(reduceSumTensor3) # n*2

        # layer 4: word2vec layer
        word2vecVariableTensor = tf.Variable(similarityVariableInitValue, name=__word2vecVariableTensorName+str(modelID)) # 1
        reluTensor = tf.nn.relu(word2vecVariableTensor * inputTensorS) # n*2
        dotTensor = reduceSumTensor3 * tf.log(tf.constant(math.e, dtype=tf.float64) + reluTensor) # n*2

        #dotTensor = reduceSumTensor3 * tf.nn.sigmoid(reluTensor)  # n*2

        # final layer: output layer
        constantTensor = tf.constant([[1.], [-1.]], dtype=tf.float64)  # larger first, smaller second
        outputTensor = tf.matmul(dotTensor, constantTensor, name=__outputTensorName+str(modelID)) # n*1

        # accuracy tensor
        equalTensor = tf.equal(tf.sign(tf.reduce_sum(outputTensor, 1)), tf.reduce_sum(inputTensorY, 1))
        accuracyTensor = tf.reduce_mean(tf.cast(equalTensor, tf.float32), name=__accuracyTensorName+str(modelID)) # 1

        # set optimizer and loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputTensor, labels=inputTensorY))
        optimizer = tf.train.AdamOptimizer(1e-4)
        min_operation = optimizer.minimize(loss, name=__finalTensorName+str(modelID))

        return min_operation

def __fit(trainModel, dataset, similarityTrainset, labels, modelID):
    length = len(dataset)
    batchTimes = int(length/__batchSize)

    with graph.as_default():
        sess.run(tf.global_variables_initializer())
    for _ in range(__epochNum):
        for batchIndex in range(batchTimes):
            indexes = np.random.choice(a=length, size=__batchSize)
            x = graph.get_tensor_by_name(__inputTensorXName+str(modelID)+":0")
            y_ = graph.get_tensor_by_name(__inputTensorYName+str(modelID)+":0")
            w = graph.get_tensor_by_name(__inputTensorWName+str(modelID)+":0")
            sess.run(trainModel, feed_dict={x: dataset[indexes], w: similarityTrainset[indexes], y_: labels[indexes]})

def __eval(testset, similarityTestset, labels, modelID):
    length = len(testset)
    batchTimes = int(length/__batchSize)

    accuracyList = []
    for _ in range(batchTimes):
        indexes = np.random.choice(a=length, size=__batchSize)
        x = graph.get_tensor_by_name(__inputTensorXName+str(modelID)+":0")
        y_ = graph.get_tensor_by_name(__inputTensorYName+str(modelID)+":0")
        w = graph.get_tensor_by_name(__inputTensorWName+str(modelID)+":0")
        outputTensor = graph.get_tensor_by_name(__outputTensorName+str(modelID)+":0")
        accuracyTensor = graph.get_tensor_by_name(__accuracyTensorName+str(modelID)+":0")
        accuracy = sess.run(accuracyTensor, feed_dict={x: testset[indexes], w: similarityTestset[indexes], y_: labels[indexes]})
        output = sess.run(outputTensor, feed_dict={x: testset[indexes], w: similarityTestset[indexes], y_: labels[indexes]})
        accuracyList.append(accuracy)
    return np.mean(np.array(accuracyList))

def __hasModels():
    for i in range(rankingDataset.__foldNum):
        curSubDir = os.path.join(modelDir, "{}{}".format(__subDirPrefix, i))
        if not os.path.exists(curSubDir):
            return False
    return True

def getWeights(modelID):
    docVariableTensor = graph.get_tensor_by_name(__docVariableTensorName+str(modelID)+":0")
    topicVariableTensor = graph.get_tensor_by_name(__topicVariableTensorName+str(modelID)+":0")
    searchVariableTensor = graph.get_tensor_by_name(__searchVariableTensorName+str(modelID)+":0")
    word2vecVariableTensor = graph.get_tensor_by_name(__word2vecVariableTensorName + str(modelID) + ":0")

    docVariable = sess.run(docVariableTensor)
    topicVariable = sess.run(topicVariableTensor)
    searchVariable = sess.run(searchVariableTensor)
    word2vecVariable = sess.run(word2vecVariableTensor)
    #print(docVariable, topicVariable, searchVariable, word2vecVariable)
    return (docVariable, topicVariable, searchVariable, word2vecVariable)
    #return (np.array([1.]*36), np.array([1.]*6), np.array([1.]*2), np.array([1.]))

__models = [None] * rankingDataset.__foldNum

if not __hasModels():
    for modelID in range(rankingDataset.__foldNum):
        __models[modelID] = __buildModel(modelID)
        _, testset, similarityTestset, trainset, similarityTrainset = rankingDataset.constructDatasetForModel(modelID)

        # load dataset and similarities for train
        dataset = [] # 18*n*2*36 -> n*2*36
        similarityset = [] # 18*n*2 -> n*2
        for indexID in range(len(trainset)):
            # topicID = rankingDataset.getTopicIDForTrain(modelID, indexID)
            for j in range(len(trainset[indexID])):
                dataset.append(trainset[indexID][j])
                similarityset.append(similarityTrainset[indexID][j])

        # train model
        print("Train the {}th model...".format(modelID))
        dataset = np.array(dataset)
        similarityset = np.array(similarityset)
        labels = [] # n*1
        for i in range(len(dataset)):
            labels.append([1.])
        labels = np.array(labels)
        __fit(trainModel=__models[modelID], dataset=dataset, similarityTrainset=similarityset, labels=labels, modelID=modelID)
        print("Success!")

        # load dataset and similarities for train
        dataset = [] # 6*n*2*18 -> n*2*18
        similarityset = [] # 6*n*2 -> n*2
        for indexID in range(len(testset)):
            # topicID = rankingDataset.getTopicIDForTrain(modelID, indexID)
            for j in range(len(testset[indexID])):
                dataset.append(testset[indexID][j])
                similarityset.append(similarityTestset[indexID][j])

        # eval model
        print("Eval the {}th model...".format(modelID))
        dataset = np.array(dataset)
        similarityset = np.array(similarityset)
        labels = [] # n*1
        for i in range(len(dataset)):
            labels.append([1.])
        labels = np.array(labels)
        accuracy = __eval(testset=dataset, similarityTestset=similarityset, labels=labels, modelID=modelID)
        print("Accuracy of the {}th model: {}".format(modelID, accuracy))

        # save model
        print("Save the {}th model...".format(modelID))
        curModelFile = "{}{}.model".format(__modelFilePrefix, modelID)
        curSubDir = os.path.join(modelDir, "{}{}".format(__subDirPrefix, modelID))
        curModelPath = os.path.join(curSubDir, curModelFile)
        with graph.as_default():
            saver = tf.train.Saver()
        saver.save(sess, curModelPath)
        print("Success!")
else:
    with graph.as_default():
        for modelID in range(rankingDataset.__foldNum):
            # load meta data of existed models
            print("Load the {}th model...".format(modelID))
            curModelFile = "{}{}.model".format(__modelFilePrefix, modelID)
            curSubDir = os.path.join(modelDir, "{}{}".format(__subDirPrefix, modelID))
            curModelPath = os.path.join(curSubDir, curModelFile)
            saver = tf.train.import_meta_graph(curModelPath+".meta")

            # load variable values of existed models
            __models[modelID] = saver.restore(sess, curModelPath)

