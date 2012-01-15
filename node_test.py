import Neural
import copy
import numpy as np
import random as r
import cv2.cv as cv

def getRandomMatrixAddress(shape):
    address=[]
    for axis_limit in shape:
        address.append(r.randint(0,axis_limit))
    return address

class NodeTest:
    """Tests nodes"""
    def __init__(self):
        self.vis=False
        self.shape=0
        self.node=Neural.Node()
        self.layer=[]

        self.matrix=[]
        self.image=[]
        self.capture=0
        self.visionSetup()

    def setVis(self,value):
        if self.vis!=value:
            self.vis=value
            cv.NamedWindow("camera", 1)

    def visionSetup(self):
        if self.vis:
            cv.NamedWindow("camera", 1)
        self.capture = cv.CaptureFromCAM(0)
        self.shape=self.getShape()

    def getShape(self):
        self.image = cv.GetMat(cv.QueryFrame(self.capture))
        n = (np.asarray(self.image)).astype(np.uint8)
        return n.shape
    
    def readFrame(self):
        self.image = cv.GetMat(cv.QueryFrame(self.capture))
        if(self.vis):
            cv.ShowImage("camera", self.image)

        self.matrix = (np.asarray(self.image)).astype(np.uint8)

    def populateFirstLayer(self,n):
        self.layer=[]
        for x in range(n):
            self.layer.append(self.makeNode())
        

    def makeNode(self):
        setattr(self.node,"mem",100)
        setattr(self.node,"size",3)
        clone=copy.deepcopy(self.node)
        for x in range(clone.size):
            self.giveNodeRandomConnection(clone)
        return clone

    def giveNodeRandomConnection(self,node):
        address=getRandomMatrixAddress(self.shape)
        node.addAddress(address)
        

test=NodeTest()
test.populateFirstLayer(2)
while 1:
    test.readFrame()
    print test.layer[1].addresses
    if cv.WaitKey(6) == 27:
        break

