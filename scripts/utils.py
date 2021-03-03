import numpy as np
import math
import random
from collections import namedtuple

# MAP CONFIGURATION
UNKNOWN = 0
VISITED = 60
FREE = 120
OBSTACLE = 240

# REWARD
COLLISION = -10000
FINISH = 10000
REVISITED = -10
ROTATION = -5
ARRIVE = 100

# STATE
INPROGRESS = -1
FAIL = 0
SUCCESS = 1

# VALUE
UNKNOWN_VAL = 0.0
VISITED_VAL = 0.1
OBSTACLE_VAL = 0.2
FREE_VAL = 0.3
POSITION_VAL = 0.7
HEADING_VAL = 0.05

Transition = namedtuple('Transition',
                        ('cur_state', 'action', 'next_state', 'reward'))
# Transition = namedtuple('Transition',
                        # ('cur_map', 'cur_pose', 'action', 'next_map', 'next_pose', 'reward'))



def getType(pixel):
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        return OBSTACLE
    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
        return FREE
    print("INVALID PIXEL VALUE")
    return -1


def dist2(a, b):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)