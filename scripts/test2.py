import cv2
import numpy as np

i = 0
while(True):
    i = (i+1) % 255
    a = [[i for j in range(300)] for k in range(300)]
    a = np.array(a, dtype=np.uint8)
    cv2.imshow('test',a)
    cv2.waitKey(1)