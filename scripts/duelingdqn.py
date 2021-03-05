import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DuelingDQN(nn.Module):

    def __init__(self, n_angle, n_actions, n_hidden1, n_hidden2):
        super(DuelingDQN, self).__init__()
        self.inputs = n_angle * 4
        self.hidden1 = n_hidden1
        self.hidden2 = n_hidden2
        self.outputs = n_actions
        self.h_layer1 = nn.Linear(self.inputs, self.hidden1)
        self.h_layer2 = nn.Linear(self.hidden1,self.hidden2)

        self.fc1 = nn.Linear(self.hidden2//2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.valuelayer = nn.Linear(64,3)
        self.actionlayer = nn.Linear(64,1)

    def forward(self, state):
        x = F.relu(self.h_layer1(state))
        x = F.relu(self.h_layer2(x))
        x1, x2 = torch.split(x, self.hidden2 // 2, dim = 1)
        # x1 = x1.unsqueeze(0)
        # x2 = x2.unsqueeze(0)
        self.value = F.relu(self.valuelayer(F.relu(self.fc2(F.relu(self.fc1(x1))))))
        self.action = F.relu(self.actionlayer(F.relu(self.fc2(F.relu(self.fc1(x1))))))
        
        self.q_val = self.value + self.action - torch.mean(self.action)

        return self.q_val