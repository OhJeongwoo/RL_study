import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):

    # def __init__(self, h, w, outputs, linear_hidden_size):
    #     super(DQN, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
    #     self.bn2 = nn.BatchNorm2d(32)
    #     self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1)
    #     self.bn3 = nn.BatchNorm2d(16)
    #     self.conv4 = nn.Conv2d(16, 8, kernel_size=5, stride=1)
    #     self.bn4 = nn.BatchNorm2d(8)

    #     # Number of Linear input connections depends on output of conv2d layers
    #     # and therefore the input image size, so compute it.
    #     def conv2d_size_out(size, kernel_size = 5, stride = 1):
    #         return (size - (kernel_size - 1) - 1) // stride  + 1
    #     convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
    #     convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
    #     linear_input_size = convw * convh * 8 + 3
    #     print(linear_input_size)
    #     self.hidden = nn.Linear(linear_input_size, linear_hidden_size)
    #     self.out = nn.Linear(linear_hidden_size, outputs)

    # # Called with either one element to determine next action, or a batch
    # # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # def forward(self, map, pose):
    #     x = F.relu(self.bn1(self.conv1(map)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu(self.bn4(self.conv4(x)))
    #     x = F.relu(self.hidden(torch.cat((x.view(x.size(0),-1),pose), dim=-1)))
    #     return self.out(x)

    def __init__(self, n_angle, n_actions, n_hidden1, n_hidden2):
        super(DQN, self).__init__()
        self.inputs = n_angle * 4
        self.hidden1 = n_hidden1
        self.hidden2 = n_hidden2
        self.outputs = n_actions
        self.h_layer1 = nn.Linear(self.inputs, self.hidden1)
        self.h_layer2 = nn.Linear(self.hidden1,self.hidden2)
        self.out = nn.Linear(self.hidden2, self.outputs)

    def forward(self, state):
        x = F.relu(self.h_layer1(state))
        x = F.relu(self.h_layer2(x))
        print(self.out(x))
        return self.out(x)