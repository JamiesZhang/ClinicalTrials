#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
from preprocess import rankingDataset

curDir = os.path.dirname(os.path.abspath(__file__))

modelDir = os.path.join(os.path.dirname(curDir), "models")
__modelFilePrefix = "MP"

__searchModelNum = 2
__topicFieldNum = 3
__docFieldNum = 3

__inputTensorXName = "inputx"
__inputTensorYName = "inputy"
__docVariableTensorName = "docVariable"
__topicVariableTensorName = "topicVariable"
__searchVariableTensorName = "searchVariable"
__finalTensorName = "train"

class RankingWeightedSumLayer(tf.keras.layers.Layer):
    def __init__(self, outputDim, **kwargs):
        self.outputDim = outputDim
        super(RankingWeightedSumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[2], ))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(RankingWeightedSumLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dotTensor = inputs * self.kernel
        reshapeTensor = tf.reshape(dotTensor, shape=(-1, 2, self.outputDim, int(inputs.shape[2].value/self.outputDim)))
        if self.outputDim == 1:
            constantTensor = tf.constant([[1.], [-1.]]) # larger first, smaller second
            return tf.matmul(tf.reduce_sum(tf.reduce_sum(reshapeTensor, 3), 2), constantTensor)
        else:
            return tf.reduce_sum(reshapeTensor, 3)

    def compute_output_shape(self, input_shape):
        if self.outputDim == 1:
            shape = tf.TensorShape((tf.TensorShape(input_shape).as_list()[0], 1))
        else:
            shape = tf.TensorShape(input_shape).as_list()
            shape[-1] = self.outputDim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(RankingWeightedSumLayer, self).get_config()
        base_config['output_dim'] = self.outputDim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def __buildModel():
    with tf.Graph().as_default():
        # compute dimensions
        docLayerDim = __searchModelNum * __topicFieldNum * __docFieldNum # 18
        topicLayerDim = __searchModelNum * __topicFieldNum # 6
        searchLayerDim = __searchModelNum # 2

        # layer 0: input layer
        inputTensorX = tf.placeholder(shape=(-1, 2, docLayerDim), name=__inputTensorXName) # n*2*18
        inputTensorY = tf.placeholder(shape=(-1, 1), name=__inputTensorYName) # n*1

        # layer 1: doc-field layer
        docVariableTensor = tf.Variable(np.random.randn(docLayerDim), name=__docVariableTensorName) # 18
        dotTensor1 = inputTensorX * docVariableTensor # n*2*18
        reshapeTensor1 = tf.reshape(dotTensor1, shape=(-1, 2, topicLayerDim, __docFieldNum)) # n*2*6*3
        reduceSumTensor1 = tf.reduce_sum(reshapeTensor1, 3) # n*2*6

        # layer 2: topic-field layer
        topicVariableTensor = tf.Variable(np.random.randn(topicLayerDim), name=__topicVariableTensorName) # 6
        dotTensor2 = reduceSumTensor1 * topicVariableTensor # n*2*6
        reshapeTensor2 = tf.reshape(dotTensor2, shape=(-1, 2, searchLayerDim, __topicFieldNum))  # n*2*2*3
        reduceSumTensor2 = tf.reduce_sum(reshapeTensor2, 3)  # n*2*2

        # layer 3: search-field layer
        searchVariableTensor = tf.Variable(np.random.randn(searchLayerDim), name=__searchVariableTensorName) # 2
        dotTensor3 = reduceSumTensor2 * searchVariableTensor  # n*2*2
        reshapeTensor3 = tf.reshape(dotTensor3, shape=(-1, 2, 1, __searchModelNum))  # n*2*1*2
        reduceSumTensor3 = tf.reduce_sum(reshapeTensor3, 3)  # n*2*1

        # final layer: output layer
        constantTensor = tf.constant([[1.], [-1.]])  # larger first, smaller second
        outputTensor = tf.matmul(tf.reduce_sum(reduceSumTensor3, 2), constantTensor) # n*1

        # set optimizer and loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputTensor, labels=inputTensorY))
        optimizer = tf.train.AdamOptimizer(1e-4)
        min_operation = optimizer.minimize(loss, name=__finalTensorName)

    return min_operation

def __fit(trainModel, sess, dataset, labels, batch_size=32, epochs=3):
    length = len(dataset)
    batchTimes = int(length/batch_size)

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batchIndex in range(batchTimes):
            indexes = np.random.choice(a=length, size=batch_size)
            x = tf.get_default_graph().get_tensor_by_name(__inputTensorXName+":0")
            y_ = tf.get_default_graph().get_tensor_by_name(__inputTensorYName + ":0")
            sess.run(trainModel, feed_dict={x: dataset[indexes], y_: dataset[indexes]})

def __hasModels():
    for i in range(rankingDataset.__foldNum):
        curModelFile = "{}{}.model".format(__modelFilePrefix, i)
        curModelPath = os.path.join(modelDir, curModelFile)
        if not os.path.exists(curModelPath):
            return False
    return True

__models = [None] * rankingDataset.__foldNum

if not __hasModels():
    with tf.Session().as_default() as sess:
        saver = tf.train.Saver()
        for modelID in range(rankingDataset.__foldNum):
            __models[modelID] = __buildModel()
            _, _, trainset = rankingDataset.constructDatasetForModel(modelID)

            # load dataset
            dataset = [] # n*2*18
            for indexID in range(len(trainset)):
                # topicID = rankingDataset.getTopicIDForTrain(modelID, indexID)
                for j in range(len(trainset[indexID])):
                    dataset.append(trainset[indexID][j])

            # train model
            print("Train the {}th model...".format(modelID))
            labels = [[1.] * len(dataset)] # n*1
            __fit(trainModel=__models[modelID], sess=sess, dataset=dataset, labels=labels, batch_size=32, epochs=3)
            print("Success!")

            # save model
            print("Save the {}th model...".format(modelID))
            curModelFile = "{}{}.model".format(__modelFilePrefix, modelID)
            curModelPath = os.path.join(modelDir, curModelFile)
            # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [__finalTensorName])
            # if tf.gfile.Exists(curModelPath):
            #     tf.gfile.Remove(curModelPath)
            # with tf.gfile.FastGFile(curModelPath, mode="wb") as fd:
            #     fd.write(constant_graph.SerializeToString())
            saver.save(sess, curModelPath)
            print("Success!")
else:
    # load existed models
    for modelID in range(rankingDataset.__foldNum):
        print("Load the {}th model...".format(modelID))
        curModelFile = "{}{}.model".format(__modelFilePrefix, modelID)
        curModelPath = os.path.join(modelDir, curModelFile)

        saver = tf.train.import_meta_graph(curModelPath+"*.meta")
        __models[modelID] = saver.restore(curModelPath)

# __model = __buildModel()
# data0 = [[i for i in range(18)],[i-1 for i in range(18)]]
# weight0 = __model.get_weights()[0]
# data1 = [[0]*6, [0]*6]
# for m in range(2):
#     for i in range(18):
#         index = int(i/3)
#         data1[m][index] = data1[m][index] + data0[m][i]*weight0[i]
# print(data0)
# print(weight0)
# print(data1)
# print()
#
# weight1 = __model.get_weights()[1]
# data2 = [[0]*2,[0]*2]
# for m in range(2):
#     for i in range(6):
#         index = int(i/3)
#         data2[m][index] = data2[m][index] + data1[m][i]*weight1[i]
# print(data1)
# print(weight1)
# print(data2)
# print()
#
# weight2 = __model.get_weights()[2]
# print(data2[0][0]*weight2[0] + data2[0][1]*weight2[1]-data2[1][0]*weight2[0] - data2[1][1]*weight2[1])
# print(__model.predict(np.array(data0).reshape([1,2,18]), batch_size=1))






