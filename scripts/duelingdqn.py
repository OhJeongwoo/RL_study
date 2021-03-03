import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

EPISODES = 1000
#after environment is made




class DuelingDQN(nn.Module):
    def __init__(self, h, w, outputs):
        self.state_size = state_size
    
    def forward(self, x):




BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

