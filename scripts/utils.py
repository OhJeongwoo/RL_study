import numpy as np

FREE = 0
OBSTACLE = 1
VISIBLE = 2

def getType(pixel):
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        return OBSTACLE
    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
        return FREE
    print("INVALID PIXEL VALUE")
    return -1
