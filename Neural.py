""" Copyright (c) 2011, Joseph M Wright
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import linalg as LA
from numpy import array, mean

import pickle
import itertools
import math
import random
import copy
import itertools
import string
#random.seed((1000,2000))

def distance(a,b): #co-domain =[0..1]
    """euclidean normalized distance between two vectors"""
    shorter=min(len(a),len(b))
    return 1-(LA.norm(a-b)/math.sqrt(shorter*1.0))

class Node:
    """A single 'neuron'"""
    def __init__(self,size,memory=1000):
        self.name   = "" #name to refer 
        self.disc = "" #description of node
        self.output = [] #output layer buffer
        self.input  = [] #input layer buffer

        self.source = 0  #source node or not
        self.file   = "" #file source to query
        self.blocksize=20 # number of chars to pull at a time

        self.size   = size #number of inputs
        self.age    = 0
        self.bias   = np.zeros([self.size+3]) #a prior prob
        self.mem    = memory # history buffer size
        self.hist   = [] # history 
        self.ohist  = [] # output history
        self.k      = [] # categories
        self.ks     = 2  # number of clusters

        self.offset = 0
        self.inmap  = [] #3-tuples of (layer,node,slot)
        self.slots  = [] #normalized inmap
        self.r      = 0.5 #introvert or extrovert
        self.margin = 0.2 #margin of error

        self.chain  = [self.updateAge, self.updateHistory, self.computeOutput, self.calcBias,  self.cluster]



    def makeNode(self,params):
        self.size=params[0]
        self.age=params[1]

    def connect(self,layer_node_slot):
        """make a connection to this node to a (layer,node,slot)
        updates inmap"""
        if layer_node_slot not in self.inmap:
            self.inmap.append(layer_node_slot)
            self.size+=1

    def __getitem__(self,n):
        # Don't depend on this 
        return self.output[n]
    
    def explain(self):
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

        # now compute expectations
    def updateAge(self):
        """Update the age of a node"""
        self.age+=1

    def computeOutput(self):
        """Compute the output vector of the node"""
        for category in self.k:
            diff = distance(category,self.input)
            self.output = diff
            self.ohist.append(diff)

 
    def cluster(self):
        """Build the categories of the node by kmeans clustering the history"""
        if len(self.k)==0 and self.age>self.mem:
            self.k.extend(kmeans(array(self.hist),self.ks)[0])

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
        if random.randint(0,self.mem)!=0:
            return 0

        # Actual computation
        self.bias=array(self.hist).mean(axis=0)
#        print self.age,self.mem,self.bias
        return 1  

#############################################
########## Layer ############################
#############################################

class Layer:
    """A single layer of nodes"""
    def __init__(self,nodenum,insize,memsize):
        self.level        = 0
        self.length       = nodenum
        self.nodes        = []
        self.output_layer = []
        self.avginput     = insize
        self.banksize     = memsize

        for x in range(nodenum):
            self.nodes.append(Node(insize,memsize))

#   def stats(self):

    def __getitem__(self,n):
        return self.nodes[n]

    def append(self,x):
        "add a Node to this layer"
        self.nodes.append(x)
        self.length += 1

    def metric(self):
        """Return the dimensions of output layer; a list of the sizes of the output layer"""

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
    def __init__(self):
        self.shape=[]
        self.layers=[]

    def __getitem__(self,n):
        return self.layers[n]

    def addLayer(self): #on top
        self.layers.append(Layer(0,20,20))
        self.layers[-1].level=len(self.layers)

    def addNode(self,level,node):
        "Add node to a certain layer"
        if level > len(self.layers)-1:
            level=len(self.layers)-1
        self[level].append(node)

    def throwNodeNormal(self,sigma,n):
        "Throw a node to watch the 2D input, with random center and spread of sigma on gauss with n inputs"
        self.addNode(1,Node(n,100))
        # assuming one input matrix for now
        slots=[]
        for limit in self[0][0].output.shape:
            center=random.randint(0,limit-1)
            slots.append([])
            for x in range(n):
                ans=int(random.gauss(center,sigma))
                if(limit<5):
                    ans=random.randint(0,limit)
                if ans < 0:
                    ans = 0
                elif ans > limit-1: 
                    ans = limit-1
                slots[-1].append(ans)
        for slot in zip(*slots):
            self[1][-1].connect([0,0,slot])

    def readMatrix(self,matrix):
        """Puts a new matrix into the source node"""
        self[0][-1].output = matrix
        
    def pullup(self,level):
        "Like pushup()"
        if level!=0:
            for node in self.layers[level]:
                values=[]
                for cn in node.inmap:
                    sl=cn[2]
                    values.append(self[cn[0]][cn[1]].output[sl[0]][sl[1]][sl[2]])
                node.readin2(values)
        
    def cycle(self):
        "Processes one epoch"
        for x in range(len(self.layers)):
            self.pushup(x)

