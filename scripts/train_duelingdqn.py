import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import yaml
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from duelingdqn import *
import environment as env
import utils

Transition = namedtuple('Transition',
                        ('cur_map', 'cur_pose', 'action', 'next_map', 'next_pose', 'reward'))

### Open yaml file ###
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
yaml_path = project_path + "/args.yaml"

with open(yaml_path) as f:
    yaml_file = yaml.load(f)
######################

### Set up hyper parameters ###
map_path = project_path + "/" + yaml_file['map_name']
visible_threshold = yaml_file['visible_threshold']
angle_interval = yaml_file['angle_interval']
step_size = yaml_file['step_size']
time_threshold = yaml_file['time_threshold']

BATCH_SIZE = yaml_file['batch_size']
GAMMA = yaml_file['gamma']
EPS_START = yaml_file['eps_start']
EPS_END = yaml_file['eps_end']
EPS_DECAY = yaml_file['eps_decay']
TARGET_UPDATE = yaml_file['target_update']

n_actions = yaml_file['actions']
n_episodes = yaml_file['episodes']

hidden_layer_size = yaml_file['hidden_layer_size']

###############################

# initialize environment
env = env.Environment(map_path, visible_threshold, angle_interval, step_size, time_threshold)
map_height, map_width = env.getSize()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = T.Compose([T.ToPILImage(),
                       T.ToTensor()])

policy_net = DuelingDQN(map_height, map_width, n_actions, hidden_layer_size).to(device)
target_net = DuelingDQN(map_height, map_width, n_actions, hidden_layer_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = utils.ReplayMemory(10000)


steps_done = 0

def select_action(map, pose):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(map,pose))
            # return policy_net(map, pose).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask_map = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_map)), device=device, dtype=torch.bool)
    non_final_next_map = torch.cat([s for s in batch.next_map if s is not None])
    non_final_next_pose = torch.cat([s for s in batch.next_pose if s is not None])
    cur_map_batch = torch.cat(batch.cur_map)
    cur_pose_batch = torch.cat(batch.cur_pose)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(cur_map_batch, cur_pose_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask_map] = target_net(non_final_next_map, non_final_next_pose).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




for i_episode in range(n_episodes):
    # Initialize the environment and state
    env.start()
    cur_map = transform(torch.from_numpy(env.getImage())).unsqueeze(0).to(device)
    cur_pose = torch.from_numpy(env.getPose()).unsqueeze(0).to(device)
    for t in count():
        # Select and perform an action
        action = select_action(cur_map, cur_pose)
        reward, done = env.doAction(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        next_map = transform(torch.from_numpy(env.getImage())).unsqueeze(0).to(device)
        next_pose = torch.from_numpy(env.getPose()).unsqueeze(0).to(device)
        if done:
            next_map = None
            next_pose = None

        # Store the transition in memory
        memory.push(cur_map, cur_pose, action, next_map, next_pose, reward)


        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            #episode_durations.append(t + 1)
            episode_durations.append(env.rewards)
            plot_durations()
            break
        cur_map = next_map
        cur_pose = next_pose
        # img = np.array(env.getImage(), dtype = np.uint8)
        # cv2.imshow('GAME',img) 
        # cv2.waitKey(1)
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()