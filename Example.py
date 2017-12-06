import Neural

# Create a network
N=Neural.Network()

# Add some nodes
N.populateFirstLayer(1000)

# Repeatedly process a frame from camera (must have OpenCV)
while 1:
    N.readFrame()
    N.pullUp()
    if Neural.cv.WaitKey(6)==27:
        break




