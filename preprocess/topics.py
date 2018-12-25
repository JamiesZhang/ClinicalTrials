#!/usr/bin/env python

import xml.dom.minidom
import json
import os

curDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.dirname(curDir), "data")

__rawTopicsFile = "topics2017.xml"
__extenedTopicsFile = "extendedTopics.xml"

sep = ','

# the class for Topic datatype (query in PM)
class Topic(object):
    def __init__(self, **kwargs):
        '''
            Create Topic instance.
            Method 1:
                data = {"number": 1, "disease": "HIV,HIV1", "gene": "KRAS,KRAS1", "demographic": "Age", "other": "other thing"}
                topic = Topic(jsonStr=data)
            Method 2:
                topic = Topic(number=1, disease="HIV,HIV1", gene="KRAS,KRAS1", demographic="Age", other="other thing")
        '''
        keysNum = len(kwargs.keys())
        if keysNum!=1 and keysNum!=5:
            raise RuntimeError("Invalid parameters while creating Topic instance! Keys num is {}.".format(keysNum))

        if 'jsonStr' in kwargs.keys():
            jsonObj = json.loads(kwargs['jsonStr'])
            self.number = int(jsonObj['number'])
            self.disease = jsonObj['disease']
            self.gene = jsonObj['gene']
            self.demographic = jsonObj['demographic']
            self.other = jsonObj['other']
        else:
            self.number = int(kwargs['number'])
            self.disease = kwargs['disease']
            self.gene = kwargs['gene']
            self.demographic = kwargs['demographic']
            self.other = kwargs['other']

    @staticmethod
    def __join(wordList):
        return sep.join(wordList)

    @staticmethod
    def __split(wordStr):
        return wordStr.split(sep)

    def getDiseaseList(self):
        return Topic.__split(self.disease)

    def getGeneList(self):
        return Topic.__split(self.gene)

    def setDiseaseList(self, diseaseList):
        self.disease = Topic.__join(diseaseList)

    def setGeneList(self, geneList):
        self.gene = Topic.__join(geneList)

    def toJsonObj(self):
        '''
            Return the json object (python dict) of this topic instance which may be used in elasticsearch module.
        '''
        jsonObj = {'number':self.number, 'disease':self.disease, 'gene':self.gene,
                   'demographic':self.demographic, 'other':self.other}
        return jsonObj

    def toJsonStr(self):
        '''
            Return the json string of this topic instance which may be used in elasticsearch module.
        '''
        jsonObj = self.toJsonObj()
        jsonStr = json.dumps(jsonObj)
        return jsonStr

# Load topics2017.xml by default including 30 Topic instances
def loadTopics(loadFile=__rawTopicsFile):
    # Read topics.xml
    loadPath = os.path.join(dataDir, loadFile)
    DOMTree = xml.dom.minidom.parse(loadPath)
    rootNode = DOMTree.documentElement
    if rootNode.tagName != "topics":
        raise RuntimeError("Invalid format of topics.xml! Tagname of root node is {}.".format(rootNode.tagName))

    # Get all of topic nodes
    topicNodes = rootNode.getElementsByTagName(name = "topic")
    topics = [None]*len(topicNodes) # List of Topic instances
    print(len(topics))
    for topicNode in topicNodes:
        number = int(topicNode.getAttribute("number"))
        disease = str(topicNode.getElementsByTagName("disease")[0].firstChild.nodeValue)
        gene = str(topicNode.getElementsByTagName("gene")[0].firstChild.nodeValue)
        demographic = str(topicNode.getElementsByTagName("demographic")[0].firstChild.nodeValue)
        other = str(topicNode.getElementsByTagName("other")[0].firstChild.nodeValue)
        topics[number-1] = Topic(number=number, disease=disease, gene=gene, demographic=demographic, other=other)
    return topics

def loadRawTopics():
    return loadTopics(__rawTopicsFile)

def loadExtendedTopics():
    return loadTopics(__extenedTopicsFile)

def hasExtendedTopics():
    extendedPath = os.path.join(dataDir, __extenedTopicsFile)
    return os.path.exists(extendedPath)

# Save topics after synonymous extended
def saveExtendedTopics(topics, saveFile=__extenedTopicsFile):
    savePath = os.path.join(dataDir, saveFile)
    if os.path.exists(savePath):
        os.remove(savePath)

    # create a xml document
    doc = xml.dom.minidom.Document()

    # create the root node
    rootNode = doc.createElement("topics")
    doc.appendChild(rootNode) # insert the root node into dom tree

    for topic in topics:
        # create and insert current topic node
        topicNode = doc.createElement("topic")
        topicNode.setAttribute("number", str(topic.number))
        rootNode.appendChild(topicNode)

        # create the child nodes of current topic node
        diseaseNode = doc.createElement("disease")
        diseaseNode.appendChild(doc.createTextNode(topic.disease))
        geneNode = doc.createElement("gene")
        geneNode.appendChild(doc.createTextNode(topic.gene))
        demographicNode = doc.createElement("demographic")
        demographicNode.appendChild(doc.createTextNode(topic.demographic))
        otherNode = doc.createElement("other")
        otherNode.appendChild(doc.createTextNode(topic.other))

        # insert the child nodes of current topic node
        topicNode.appendChild(diseaseNode)
        topicNode.appendChild(geneNode)
        topicNode.appendChild(demographicNode)
        topicNode.appendChild(otherNode)

    with open(savePath, 'wb') as fp:
        fp.write(doc.toprettyxml(encoding='utf-8'))


