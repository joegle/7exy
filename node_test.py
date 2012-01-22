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

def getNormalMatrixAddress(shape,n):
    """Throw normal"""
    limit=4
    std=2
    address=[]
    center=[]
    for axi in shape:
        center.append(r.randomint(0,axi-1))
    address.append(center)
    c=[]
    x=0
    for axi in shape:
        if axi<=limit:
            c.append(np.random.randint(0,axi-1,n-1))
        else:
            c.append(np.random.normal(center[x],std,n-1))
        x+=1

class NodeTest:
    """Tests nodes"""
    def __init__(self):
        self.vis=False
        self.back=True
        self.shape=0
        self.node=Neural.Node()
        self.layer=[]


        self.matrix=[]
        self.image=[]
        self.capture=0
        self.visionSetup()
        self.backprojection=np.zeros(self.shape)

    
    def setVis(self,value):
        if self.vis!=value:
            self.vis=value
            cv.NamedWindow("camera", 1)


    def visionSetup(self):
        if self.vis:
            cv.NamedWindow("camera", 1)
        if self.back:
            cv.NamedWindow("back", 1)
        self.capture = cv.CaptureFromCAM(0)
        self.shape=self.getShape()
        self.backprojection=np.zeros(self.shape)

    def backProject(self):
        self.backprojection=np.zeros(self.shape).astype(np.uint8)
        for node in self.layer:
            self.backProjectNode(node)


    def backProjectNode(self,node):
        #print node.addresses
        for a,b in zip(node.addresses,node.input):
            self.backprojection[a[0]][a[1]][a[2]]=b
        #self.backprojection.put(np.array(node.addresses),np.array(node.input))

    def getShape(self):
        self.image = cv.GetMat(cv.QueryFrame(self.capture))
        n = (np.asarray(self.image)).astype(np.uint8)
        return n.shape
    
    def readFrame(self):
        self.image = cv.GetMat(cv.QueryFrame(self.capture))
        if(self.vis):
            cv.ShowImage("camera", self.image)

        if self.back:            
            self.backProject()
            #cv.SetData( cv_im, a.tostring(),  a.dtype.itemsize * nChannels * a.shape[1] )            
            #1920
            #mat=cv.CreateMat(self.shape[ 0 ],self.shape[1],cv.CV_8UC3)
            #mat=cv.fromarray(self.backprojection)
            #img=cv.CreateImage((self.shape[1],self.shape[0]),8,3)
            #cv.SetData(img,self.backprojection,1920)
            cv.ShowImage("back",cv.fromarray(self.backprojection,False))
            
        self.matrix = (np.asarray(self.image)).astype(np.uint8)

    def populateFirstLayer(self,n):
        self.layer=[]
        for x in range(n):
            self.layer.append(self.makeNode())

    def addNodeToFirstLayer(self,node):
        self.layer.append(node)
        

    def makeNode(self):

        setattr(self.node,"mem",100)
        
        s=max(2,int(np.random.normal(4,2)))
        setattr(self.node,"size",s)

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

    def pullUp(self):
        """ First layer nodes collect and process their data"""
        for node in self.layer:
            data = self.getNodeData(node)
            node.readin2(data)
#        self.f()
            

    def f(self):
        for node in self.layer:
            print node.bias,
        print
        
        
node_num=1500
test=NodeTest()
test.populateFirstLayer(node_num)

while 1:
    test.readFrame()
    test.pullUp()
#    a= test.layer[0].addresses[1]
#    print a,test.shape
#    print test.getElement(a)
#    print test.getNodeData(test.layer[0])
    if cv.WaitKey(6) == 27:
        break

