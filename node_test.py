import Neural
import copy
import numpy as np
import random as r
import cv2.cv as cv

def getRandomMatrixAddress(shape):
    address=[]
    for axis_limit in shape:
        address.append(r.randint(0,axis_limit-1))
#    print shape,address
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

    def getElement(self,address):
        return self.matrix.item(tuple(address))

    def getNodeData(self,node):
        data=[]
        for address in node.addresses:
            ans=self.getElement(address)
            data.append(ans)
        return data
    
    def f(self):
        for node in self.layer:
            print self.getNodeData(node)
        
        

test=NodeTest()
test.populateFirstLayer(2)
while 1:
    test.readFrame()
    test.f()
#    a= test.layer[0].addresses[1]
#    print a,test.shape
#    print test.getElement(a)
#    print test.getNodeData(test.layer[0])
    if cv.WaitKey(6) == 27:
        break

