""" Copyright (c) 2011, Joseph M Wright
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import random as r
from scipy.cluster.vq import kmeans2
from numpy import linalg as LA
from numpy import array, mean
import cv2.cv as cv


import pickle
import itertools
import math
import copy
import itertools
import string
#random.seed((1000,2000))

def distance(a,b): #co-domain =[0..1]
    """euclidean normalized distance between two vectors"""
    shorter=min(len(a),len(b))
    return 1-(LA.norm(a-b)/math.sqrt(shorter*1.0))

def getRandomMatrixAddress(shape):
    """Return a random address to a matrix of the shape"""
    address=[]
    for axis_limit in shape:
        address.append(r.randint(0,axis_limit-1))
    return address

def tri(m, orientation = 0):
    """get indices of corners of triangle in a square matrix, orientation is 0..3"""
    n = m.shape[0] - 1
    mid = n / 2
    m_corners = [(0,0), (0,n), (n,n), (n,0), (0,0)]
    mid_points = [(n, mid), (mid,0), (0,mid), (mid,n)]

    g = []

    g.append(m_corners[orientation])
    g.append(m_corners[orientation + 1])
    g.append(mid_points[orientation])

    return g

def hex(m, orientation = 0):
    """get indices of vertices of hexagon in a square matrix, orienation is 0..5"""
    n = m.shape[0] - 1
    mid = n / 2
    third = int(round(n * 0.3333))
    two = int(round(n* 0.6666))

    g = [(0, third), (0, two) ]

    g.append((mid, n))
    g.append((n, two))
    g.append((n, third))
    g.append((mid,0))

    g  = g[orientation:] + g[:orientation]

    return g

class Node:
    """A single 'neuron'"""
    def __init__(self):
        self.name   = "" #name to refer 
        self.discription = "" #description of node
        self.output = [0] #output layer buffer
        self.input  = [] #input layer buffer

        self.source = 0  #source node or not
        self.file   = "" #file source to query
        self.blocksize=20 # number of chars to pull at a time

        self.size   = 2 #number of inputs
        self.age    = 0
        self.bias   = np.zeros([self.size+3]) #a prior prob
        self.obias  = np.zeros([self.size+3]) #a prior prob
        self.mem    = 500 # history buffer size
        self.hist   = [] # history 
        self.ohist  = [] # output history
        self.k      = [] # categories
        self.ks     = 2  # number of clusters

        self.inmap  = [] #3-tuples of (layer,node,slot)
        self.slots  = [] #normalized inmap
        self.r      = 0.5 #introvert or extrovert
        self.margin = 0.2 #margin of error

        self.addresses = []

        self.chain  = [self.updateAge, self.updateHistory, self.computeOutput, self.calcBias,  self.cluster]


    def show(self):
        print "Size: ",self.size, " Age: ",self.age
        print "Mem: ",self.mem
        print "Bias:  ",self.bias
        print "OBias: ",self.obias
        print "Inmap:"
        for x in self.inmap:
            print x
        print "Categories:"
        for x in self.k:
            print x
        print "Output:"
        for x in self.output:
            print x


    def __getitem__(self,n):
        # Don't depend on this 
        return self.output[n]

    
    def connect(self,layer_node_slot):
        """make a connection to this node to a (layer,node,slot)
        updates inmap"""
        if layer_node_slot not in self.inmap:
            self.inmap.append(layer_node_slot)
            self.size=len(self.inmap)

    def addAddress(self,address):
        self.addresses.append(address)
        self.size=len(self.addresses)

    def explain(self):
        """ show the order of the readin call chain and whats going on"""
        n=1
        for function in self.chain:
            print str(n)+". "+function.__name__
            print "\t"+function.__doc__+"\n"
            n+=1
    
    def readin2(self,x):
        """update input and do everything in the chain process"""
        self.input=x[0:self.size]
        for function in self.chain:
            function()
         
    def readin(self,x):
        """process one input, takes an array that equals the input array"""
        self.input=x[0:self.size]
        self.updateAge()
        self.updateHistory()
        self.computeOutput()
        self.calcBias()
        self.cluster()

    def updateAge(self):
        """Update the age of a node"""
        self.age+=1

    def computeOutput(self):
        """Compute the output vector of the node"""
        self.output=[]
        for category in self.k:
            diff = distance(category,self.input)
            self.output.append(diff)
            self.ohist.append(diff)
 
    def cluster(self):
        """Build the categories of the node by kmeans clustering the history"""
        if len(self.k)==0 and self.age>self.mem:
            a = array(self.hist).astype(np.float)
            means = kmeans2(a, self.ks, minit='points')
            self.k.extend(means[0])

    def updateHistory(self):
        """Add the current input to the history"""
        # Condition
        if(0):
            return 0

       #append to history
        if len(self.hist)>=self.mem:#add to history
            self.hist.pop(0)
        self.hist.append(self.input)       
        return 1

    def calcBias(self):
        """Update the bias vector of expected value for inputs"""

        # Condition to satisfy in order to update
        if r.randint(0,self.mem)==0:
            self.bias=array(self.hist).mean(axis=0)
            self.obias=array(self.ohist).mean(axis=0)

#############################################
########## Layer ############################
#############################################

class Layer:
    """A single layer of nodes"""
    def __init__(self):
        self.level        = 0
        self.length       = 0
        self.nodes        = []
        self.output_layer = []
        self.avginput     = 0
        self.banksize     = 0

        for x in range(0):
            self.nodes.append(Node(insize,memsize))

#   def stats(self):

    def __getitem__(self,n):
        return self.nodes[n]

    def append(self,x):
        "add a Node to this layer"
        self.nodes.append(x)
        self.length += 1

    def metric(self):
        """compute the demesion of the output_layer"""
        self.output_layer=[]
        for node in self.nodes:
            self.output_layer.append(len(node.output))
        
    def giveConnection(self):
        node=r.randint(0,self.length-1)
        slot=r.randint(0,len(self.nodes[node].k))
        return [node, slot]

    def addBlankNode(self,memory=1000):
        self.length+=1
        self.nodes.append(Node(0,memory))
        
    def killNode(self,x):
        self.length-=1
        del self.nodes[x]

#####################################################
############### Network #############################
#####################################################

        
class Network:
    """Tests nodes"""
    def __init__(self):
        self.vis=False
        self.back=True
        self.shape=0
        self.node=Node()
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
        for a,b in zip(node.addresses,node.bias):
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
