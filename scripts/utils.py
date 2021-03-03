import numpy as np
import math

# MAP CONFIGURATION
UNKNOWN = 0
VISITED = 1
FREE = 2
OBSTACLE = 3

# REWARD
COLLISION = -10000
FINISH = 10000
REVISITED = -10
ROTATION = -10
ARRIVE = 100

# STATE
INPROGRESS = -1
FAIL = 0
SUCCESS = 1

def getType(pixel):
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        return OBSTACLE
    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
        return FREE
    print("INVALID PIXEL VALUE")
    return -1


def dist2(a, b):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])