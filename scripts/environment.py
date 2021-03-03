import random
import numpy as np
from PIL import Image
import cv2
from utils import *
import utils




class Environment:
    def __init__(self, map_path, visible_threshold = 5.0, angle_interval = 60, step_size = 0.2, time_threshold = 1000):
        self.visible_threshold = visible_threshold
        self.angle_interval = angle_interval
        self.step_size = step_size
        self.time_threshold = time_threshold
        self.dx = [-1, 0, 1, 0]
        self.dy = [0, -1, 0, 1]
        img = cv2.imread(map_path, cv2.IMREAD_COLOR)
        self.height, self.width, _ = img.shape
        self.map = [[0 for i in range(self.width)] for j in range(self.height)]
        for i in range(self.height):
            tmp = []
            for j in range(self.width):
                t = getType(img[i,j])
                if t == OBSTACLE:
                    self.map[i][j] = OBSTACLE
                elif t == FREE:
                    self.map[i][j] = FREE
                else:
                    print("INVALID PIXEL VALUE")
        

    def start(self):
        # start new game, initialization
        self.visible = [[False for i in range(self.width)] for j in range(self.height)]
        self.visited = [[0 for i in range(self.width)] for j in range(self.height)]
        self.time = 0
        self.rewards = 0
        self.spaces = 0
        while(True):
            x = random.randint(0, self.height-1)
            y = random.randint(0, self.width-1)
            if self.map[x][y] == OBSTACLE:
                continue
            self.position = [x,y]
            break
        self.heading = random.randint(0,3) # 0 up 1 left 2 down 3 right
        self.update()

    def update(self):
        reward = 0
        if self.visited[self.position[0]][self.position[1]] > 0:
            reward = reward + REVISITED * self.visited[self.position[0]][self.position[1]]

        else:
            self.spaces = self.spaces + 1
            reward = reward + int(ARRIVE * math.sqrt(self.spaces))
        self.visited[self.position[0]][self.position[1]] = self.visited[self.position[0]][self.position[1]] + 1
        
        for i in range(self.angle_interval):
            dx = self.step_size * math.cos(2 * math.pi * i / self.angle_interval)
            dy = self.step_size * math.sin(2 * math.pi * i / self.angle_interval)
            step = 0
            while(True):
                if step * self.step_size > self.visible_threshold:
                    break
                nx = int(self.position[0] + dx * step)
                ny = int(self.position[1] + dy * step)
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    break
                self.visible[nx][ny] = True
                if self.map[nx][ny] == OBSTACLE:
                    break 
                step = step + 1
        
        return reward

    def isEnd(self):
        if self.map[self.position[0]][self.position[1]] == OBSTACLE:
            return FAIL, True
        
        done = True
        for i in range(self.height):
            if not done:
                break
            for j in range(self.width):
                if self.map[i][j] != OBSTACLE and self.visited[i][j] == 0:
                    done =  False
                    break
        if done:
            return SUCCESS, True
        
        if self.time > self.time_threshold:
            return FAIL, True

        return INPROGRESS, False

    
    def doAction(self, action):
        # do action
        # 0 : go, 1 : turn ccw, 2 : turn cw
        self.time = self.time + 1
        reward = 0
        if action == 0:
            self.position[0] = self.position[0] + self.dx[self.heading]
            self.position[1] = self.position[1] + self.dy[self.heading]
            reward = reward + self.update()
        elif action == 1:
            self.heading = (self.heading + 1) % 4
            reward = reward + ROTATION + self.update()
        elif action == 2:
            self.heading = (self.heading + 3) % 4
            reward = reward + ROTATION + self.update()
        success, done = self.isEnd()
        
        if success == FAIL:
            reward = reward + COLLISION
        elif success == SUCCESS:
            reward = reward + FINISH
        self.rewards = self.rewards + reward
        return reward, done


    def getSize(self):
        return self.height, self.width

    def getImage(self):
        rt = [[0.0 for i in range(self.width)] for j in range(self.height)]

        for i in range(self.height):
            for j in range(self.width):
                if not self.visible[i][j]:
                    rt[i][j] = UNKNOWN_VAL
                    continue
                if self.visited[i][j] > 0:
                    rt[i][j] = VISITED_VAL
                    continue
                if self.map[i][j] == OBSTACLE:
                    rt[i][j] = OBSTACLE_VAL
                else:
                    rt[i][j] = FREE_VAL

        rt[self.position[0]][self.position[1]] = POSITION_VAL + HEADING_VAL * self.heading
        # print(self.position)
        # print(self.heading)
        # print(self.height, self.width)
        # print(self.map[self.position[0]][self.position[1]])
        return np.array(rt, dtype=np.float32)

    def getPose(self):
        return np.array([self.position[0]/self.height, self.position[1]/self.width, self.heading/4.0], dtype=np.float32)

    def getReward(self):
        return self.rewards