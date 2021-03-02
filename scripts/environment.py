import random
import numpy as np
from PIL import Image
import cv2
import utils




class Environment:
    def __init__(self, map_path):
        image = cv2.imread(map_path, cv2.IMREAD_COLOR)
        self.map = []
        self.width, self.height, _ = image.shape
        for i in range(self.width):
            tmp = []
            for j in range(self.height):
                if(image[i,j])


    def start(self):
        # start new game, initialization
        while(True):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if self.map[x][y] == OBSTACLE:
                continue
            break

    def update(self):


    def doAction(self, action):
        # do action

    def getSize(self):
        return self.width, self.height

    def getImage(self):
