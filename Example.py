import Neural
import random as r

N=Neural.Network()

#add a source(and a node)
N.addSource("data/Canyon.html",5)

N.addLayer() #layer 1 (bottom layer is 0


period=5000
population=200
for x in range(population):
    N.attachNode(1,r.randrange(3,6))

N.addLayer() #layer 1 (bottom layer is 0

for x in range(population):
    N.attachNode(1,r.randrange(3,6))

for x in range(period):
    N.cycle()


