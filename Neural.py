""" Copyright (c) 2011, Joseph M Wright
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

#import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import linalg as LA
from numpy import array, mean


import itertools
import math
import random
import copy
import itertools
import string
random.seed((1000,2000))

#TODO:
# category maker (+redundancy remover, tension splitter
# back propagation :: display expectations
# high/low finder don't know 
# Bayesian thing

def distance(a,b): #co-domain =[0..1]
    """euclidean normalized distance between two vectors"""
    shorter=min(len(a),len(b))
    return 1-(LA.norm(a-b)/math.sqrt(shorter*1.0))

    # this was the old way
    #  there is no size checking anymore
    shorter=min(len(a),len(b))
    s=0
    for x in range(shorter):
        s+= (a[x]-b[x])**2
    return 1-math.sqrt(s/(shorter*1.0))

class Node:
    """A single 'neuron'"""
    def __init__(self,size,memory=200):
        self.name   = "" #name to refer 
        self.disc = "" #description of node
        self.output = [] #output layer buffer
        self.input  = [] #input layer buffer

        self.source = 0  #source node or not
        self.file   = "" #file source to query
        self.blocksize=20 # number of chars to pull at a time

        self.age    = 0
        self.bias   = [] #a prior prob
        self.mem    = memory # history buffer size
        self.hist   = [] #history 
        self.k      = [] #categories
        self.size   = size #number of inputs
        self.offset = 0
        self.inmap  = [] #3-tuples of (layer,node,slot)
        self.slots  = [] #normalized inmap
        self.r      = 0.5 #introvert or extrovert
        self.margin = 0.2 #margin of error

    def makesource(self,source,blocksize=20):
        "Make this node read in n chars from a file"
        self.source = 1
        self.file = source
        self.blocksize=blocksize
        self.fstream=open(self.file,'r')

    def connect(self,layer_node_slot):
        """make a connection to this node to a (layer,node,slot)
        updates inmap"""
        if layer_node_slot not in self.inmap:
            self.inmap.append(layer_node_slot)
            self.size+=1

    def pullblock(self):
        """Pull one block of chars into node
        self.output becomes these values"""
        self.output=[]
        for x in range(self.blocksize):
            try:
                char=ord(self.fstream.read(1))
            except TypeError:
                break
            s=[]
            for b in [0,1,2,3,4,5,6,7]:
                s.append((char>>b)&1)
            self.output.extend(s)
        self.input=self.output
           
    def __getitem__(self,n):
        return self.output[n]
         
    def readin(self,x):
        """process one input, takes an array that equals the input array"""
        self.age+=1
        self.output=[]
        self.input=x[0:self.size]


        self.updateHistory(self.input)

        # now compute output
        for cat in self.k:
            self.output.append(distance(cat,self.input))

        self.calcBias()
        self.cluster()
        # now compute expectations

    def cluster(self,n=4):
        if len(self.k)==0 and random.randint(0,self.mem)==0:
            self.k.extend(kmeans(array(self.hist),n)[0])

        
    def updateHistory(self,x):
        """Add this term to the history"""
        # Condition
        if(0):
            return 0

       #append to history
        if len(self.hist)>=self.mem:#add to history
            self.hist.pop(0)
        self.hist.append(x)       
        return 1


    def formInmap2(self,spread,scale):
        """spread (0-1) zero is random;1 is linear, scale is the size of the frame the node is on"""
        print "lol-formInmap2" 

    def addCat(self,x):
        self.k.append(x)

    def formCats(self,n):
        self.k.extend(random.sample(self.hist,n))

    def calcBias(self):
        """Update the bias vector of expected value for inputs"""

        # Condition to satisfy in order to update
        if random.randint(0,self.mem)!=0:
            return 0

        # Actual computation
        self.bias=array(self.hist).mean(axis=0)
        print self.age,self.mem,self.bias
        return 1  

    def show(self):
        print"============"
        print "Inputs:",self.inmap,"or",self.slots
        print "BIAS:",
        for x in self.bias:
            print round(x,3),
        print "\n\nCATEGORIES:"
        for x in self.k:
            print x
        print "\nCurrent:"
        print self.input
        print "\n\nHISTORY (last 10)"
        for x in self.hist[:10]:
            print x
        print "\n\nOUTPUT"
        for x in self.output:
            print round(x,3),
        print 

#############################################
########## Layer ############################
#############################################

def topo(seq):
    """converts node inmaps to regular indexes"""
    first=seq[0]
    k=[first]
    prev=0
    for x in seq[1:]:
        k.append(x+k[prev])
    return k


class Layer:
    """A single layer of nodes"""
    def __init__(self,nodenum,insize,memsize):
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
        self.nodes.append(x)
        self.length += 1

    def metric(self):
        """Return the dimensions of output layer; a list of the sizes of the output layer"""
        bytesize=8
        size=[]
        for node in self.nodes:
            if(node.source==0):
                size.append(len(node.k))
            else:
                size.append(bytesize*node.blocksize)
        return size

    def addBlankNode(self):
        self.length+=1
        self.nodes.append(Node(0))
        
    def killNode(self,x):
        self.length-=1
        del self.nodes[x]

    def output(self):
        self.output_layer=[]
        for x in self.nodes:
            self.output_layer.append(x.output)
        return self.output_layer

    def showNodes(self,n=-1):
        #make this more interactive
        if n==-1:
            for x in self.nodes:
                x.show()
        else:
            self.nodes[n].show()

    def showTopology(self):
        """returns topology of layer"""
            
    def show(self):
        spread=100
        print "="*10,
        print "LAYER DATA",
        print "="*10,
        print self.length,"Nodes"
        print "  Ins Mem Map"
        k=0
        al="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
        for x in self.nodes:
            print "{3} {0:3d} {2:3d} {1}".format(x.size,x.inmap,x.mem,al[k])
            k+=1


#####################################################
############### Network #############################
#####################################################

def tr(seq):
    return map(lambda x: isinstance(x,float) and str(round(x,2)) or tr(x), seq)

def fairChoice(metric):
    "Return a random address from a metric"
    s=sum(metric)
    slot=random.randint(0,s-1)
    n=0
    while(metric[n]<=slot):
        slot-=metric[n]
        n+=1
    return [n,slot]

class Network:
    def __init__(self):
        self.layers=[]

    def __getitem__(self,n):
        return self.layers[n]

    def addLayer(self): #on top
        self.layers.append(Layer(0,20,20))

    def addNode(self,level):
        "Add node to a certain layer"
        if level > len(self.layers)-1:
            level=len(self.layers)-1
        self[level].addBlankNode()
        
    def addSource(self,src_file,blocksize):
        if len(self.layers)==0:
            self.addLayer()
        self[0].append(Node(0,0))
        self[0][-1].makesource(src_file,blocksize)

    def show(self):
        level=len(self.layers)-1
        for layer in reversed(self.layers):
#            print layer
            print level,
            for x in layer.metric():
                print "|"*x,'-',
            level-=1
            print
        print 

    def graph(self):
        #dot -Tpng graph.dot > output.png
        f=open("network.dot",'w')
        f.write("digraph g {\n")
        f.write("""rankdir="TB";\n""")
        f.write("""ranksep="5.0 equally";\n""")

        f.write("""node [front = "16" shape="ellipse"];\n""")
        f.write("edge [];\n")
        l=0
        n=0
        for layer in self.layers:
            n=0
            f.write("""subgraph cluster_"""+str(l)+"""first {
	 rank=same;
	 label=" """+str(l) +"""";\n """)

            for node in layer.nodes:
                f.write(""" "n"""+str(l)+str(n)+"""" \n""")
                f.write("""  [label = " """)
                f.write("""{{""")
                top=len(node.k)
                if node.source==1:
                    top=8*node.blocksize
                for x in range(top):
                    k=""
                    if x%8==0:
                        k="8"
                    f.write("<"+str(x)+">"+str(k)+"")
                    if top-1>x:
                        f.write("|")
                f.write("}|{")
                for x in range(node.size):
                    f.write("<b"+str(x)+">"+str("")+"")
                    if node.size-1>x:
                        f.write("|")
                f.write(""" }}" shape="record" color="blue"]; \n""")
                n+=1
            f.write("}\n")
            l+=1
        l=0
        colors=["cyan","blue","crimson","gold","lawngreen","indigo","purple","pink"]
        for layer in self.layers:
            n=0
            for node in layer:
                k=0
                for m in node.inmap:                
                    f.write(""" "n%s%s":b%s -> "n%s%s":%s [color="%s"];\n""" 
                            % (l,n,k,m[0],m[1],m[2],colors[random.randint(0,len(colors)-1)]))
                    k+=1
                n+=1
            l+=1
        f.write("}")
        f.close()

    def pushup(self,level):
        "Sends input up to the next layer"
        if level==0:
            for node in self.layers[0]:
                node.pullblock()
                return 1
        for node in self.layers[level]:#####
            tmp=[]
            for x in node.inmap:
                tmp.append(self[x[0]][x[1]][x[2]])
            node.readin(tmp)
        return 1

    def cycle(self):
#       "Processes one epoch"
        for x in range(len(self.layers)):
            self.pushup(x)
#            print x,self[x].output()####

    def attachNode(self,layer,number_of_inputs):
        "Adds a node to a non-source layer with input random connections"
        if layer:
            self.addNode(layer)
            metric=self[layer-1].metric()
            s=sum(metric)
            for x in range(number_of_inputs):
                connection=[layer-1]+fairChoice(metric)
                while(connection in self[layer][-1].inmap):
#                    print connection
                    connection=[layer-1]+fairChoice(metric)
                self[layer][-1].connect(connection)

