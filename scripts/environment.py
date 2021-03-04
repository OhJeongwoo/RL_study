import random
import numpy as np
from PIL import Image
import cv2
from utils import *
import utils




class Environment:
    def __init__(self, map_path, visible_threshold = 5.0, n_angle = 60, step_size = 0.2, time_threshold = 1000):
        self.visible_threshold = visible_threshold
        self.n_angle = n_angle
        self.step_size = step_size
        self.time_threshold = time_threshold
        self.free_spaces = 0
        self.dx = [-1, 0, 1, 0]
        self.dy = [0, -1, 0, 1]
        img = cv2.imread(map_path, cv2.IMREAD_COLOR)
        self.height, self.width, _ = img.shape
        self.max_distance = math.hypot(self.height, self.width)
        self.map = [[0 for i in range(self.width)] for j in range(self.height)]
        for i in range(self.height):
            tmp = []
            for j in range(self.width):
                t = getType(img[i,j])
                if t == OBSTACLE:
                    self.map[i][j] = OBSTACLE
                elif t == FREE:
                    self.map[i][j] = FREE
                    self.free_spaces = self.free_spaces + 1
                else:
                    print("INVALID PIXEL VALUE")
        

    def start(self):
        # start new game, initialization
        self.visible = [[False for i in range(self.width)] for j in range(self.height)]
        self.visited = [[0 for i in range(self.width)] for j in range(self.height)]
        self.time = 0
        self.rewards = 0
        self.spaces = 0
        self.count = 0
        self.type = True
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
            if self.type:
                self.count = 1
                self.type = False
            else:
                self.count = self.count + 1
            reward = reward + REVISITED * self.count

        else:
            self.spaces = self.spaces + 1
            if self.type:
                self.count = self.count + 1
            else:
                self.count = 1
                self.type = True
            reward = reward + ARRIVE * self.free_spaces / (self.free_spaces - self.spaces)
        self.visited[self.position[0]][self.position[1]] = self.visited[self.position[0]][self.position[1]] + 1
        
        for i in range(self.n_angle):
            dx = self.step_size * math.cos(2 * math.pi * i / self.n_angle)
            dy = self.step_size * math.sin(2 * math.pi * i / self.n_angle)
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
        return np.array(rt, dtype=np.float32)

    def getLidar(self):
        free_lidar = []
        obstacle_lidar = []
        visited_lidar = []
        unknown_lidar = []

        
        # For free space
        for i in range(self.n_angle):
            dx = self.step_size * math.cos(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            dy = self.step_size * math.sin(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            step = 0
            valid = True
            while(True):
                step = step + 1
                nx = int(self.position[0] + step * dx)
                ny = int(self.position[1] + step * dy)
                if nx == self.position[0] and ny == self.position[1]:
                    continue
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    valid = False
                    break
                if self.visible[nx][ny] and self.visited[nx][ny] ==0 and self.map[nx][ny] == FREE:
                    break

            if valid:
                free_lidar.append(step * self.step_size / self.max_distance)
            else:
                free_lidar.append(-1)
        
        # For obstacle space
        for i in range(self.n_angle):
            dx = self.step_size * math.cos(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            dy = self.step_size * math.sin(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            step = 0
            valid = True
            while(True):
                step = step + 1
                nx = int(self.position[0] + step * dx)
                ny = int(self.position[1] + step * dy)
                if nx == self.position[0] and ny == self.position[1]:
                    continue
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    valid = False
                    break
                if self.visible[nx][ny] and self.visited[nx][ny] ==0 and  self.map[nx][ny] == OBSTACLE:
                    break
                
            if valid:
                obstacle_lidar.append(step * self.step_size / self.max_distance)
            else:
                obstacle_lidar.append(-1)
        
        # For visited space
        for i in range(self.n_angle):
            dx = self.step_size * math.cos(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            dy = self.step_size * math.sin(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            step = 0
            valid = True
            while(True):
                step = step + 1
                nx = int(self.position[0] + step * dx)
                ny = int(self.position[1] + step * dy)
                if nx == self.position[0] and ny == self.position[1]:
                    continue
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    valid = False
                    break
                if self.visible[nx][ny] and self.visited[nx][ny] > 0:
                    break
                
            if valid:
                visited_lidar.append(step * self.step_size / self.max_distance)
            else:
                visited_lidar.append(-1)
        
        # For unknown space
        for i in range(self.n_angle):
            dx = self.step_size * math.cos(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            dy = self.step_size * math.sin(2 * math.pi * i / self.n_angle + self.heading * math.pi / 2 + math.pi)
            step = 0
            valid = True
            while(True):
                step = step + 1
                nx = int(self.position[0] + step * dx)
                ny = int(self.position[1] + step * dy)
                if nx == self.position[0] and ny == self.position[1]:
                    continue
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    valid = False
                    break
                if not self.visible[nx][ny]:
                    break
                
            if valid:
                unknown_lidar.append(step * self.step_size / self.max_distance)
            else:
                unknown_lidar.append(-1)
        
        free_lidar = np.array(free_lidar, dtype=np.float32)
        obstacle_lidar = np.array(obstacle_lidar, dtype=np.float32)
        visited_lidar = np.array(visited_lidar, dtype=np.float32)
        unknown_lidar = np.array(unknown_lidar, dtype=np.float32)

        return np.concatenate((free_lidar, obstacle_lidar, visited_lidar, unknown_lidar), axis=None)

    def getPose(self):
        return np.array([self.position[0]/self.height, self.position[1]/self.width, self.heading/4.0], dtype=np.float32)

    def getReward(self):
        return self.rewards

    def getSummary(self):
        return self.rewards, self.spaces, self.time