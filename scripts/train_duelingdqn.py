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

import dqn
import environment as env
import utils

rewards = []
coverages = []

Transition = namedtuple('Transition',
                        ('cur_state', 'action', 'next_state', 'reward'))

### Open yaml file ###
project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
yaml_path = project_path + "/args.yaml"


with open(yaml_path) as f:
    yaml_file = yaml.load(f)
######################

### Set up hyper parameters ###
map_path = project_path + "/" + yaml_file['map_name']
log_path = project_path + "/logs/dueling/" + yaml_file['log_file_name']
visible_threshold = yaml_file['visible_threshold']
n_angle = yaml_file['n_angle']
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

hidden_layer1_size = yaml_file['hidden_layer1_size']
hidden_layer2_size = yaml_file['hidden_layer2_size']


###############################

# initialize environment
env = env.Environment(map_path, visible_threshold, n_angle, step_size, time_threshold)
map_height, map_width = env.getSize()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs = open(log_path, 'w')

transform = T.Compose([T.ToPILImage(),
                       T.ToTensor()])

policy_net = dqn.DQN(n_angle, n_actions, hidden_layer1_size, hidden_layer2_size).to(device)
target_net = dqn.DQN(n_angle, n_actions, hidden_layer1_size, hidden_layer2_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = utils.ReplayMemory(10000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_rewards(episodes, rewards):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.plot(episodes, rewards)

    plt.savefig("./reward_graph.png")

def plot_coverage(episodes, coverage):
    plt.figure(3)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Coverage')

    plt.plot(episodes, coverage)

    plt.savefig("./coverage_graph.png")    


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    cur_state_batch = torch.cat(batch.cur_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(cur_state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0].detach()

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
    #cur_state = transform(torch.from_numpy(env.getLidar())).unsqueeze(0).to(device)
    cur_state = torch.from_numpy(env.getLidar()).unsqueeze(0).to(device)
    for t in count():
        # Select and perform an action
        action = select_action(cur_state)
        reward, done = env.doAction(action)
        reward = torch.tensor([reward], device=device)
        
        
        # Observe new state
        #next_state = transform(torch.from_numpy(env.getLidar())).unsqueeze(0).to(device)
        next_state = torch.from_numpy(env.getLidar()).unsqueeze(0).to(device)
        if done:
            next_state = None

        # Store the transition in memory
        memory.push(cur_state, action, next_state, reward)


        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            #episode_durations.append(t + 1)
            cur_rewards, spaces, durations = env.getSummary()
            episode_durations.append(env.rewards)
            data = "{0} {1} {2} {3}\n".format(i_episode,durations,cur_rewards,spaces)
            logs.write(data)
            coverage = env.spaces/env.free_spaces * 100
            print("iteration {0}, duration : {1}, rewards : {2}, coverage: {3}, free_spaces: {4}".format(i_episode,t+1,env.rewards, coverage, env.spaces))
            rewards.append(env.rewards)
            coverages.append(coverage)
            episodes = list(range(i_episode+1))
            plot_rewards(episodes, rewards)
            plot_coverage(episodes, coverages)
            #plot_durations()
            break
        else:
            img = np.array(env.getImage() * 255, dtype = np.uint8)
            cv2.imshow('GAME',img) 
            cv2.waitKey(1)
        
        cur_state = next_state


        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), './duelingdqn/policy/test0304.pkl')
        torch.save(target_net.state_dict(), './duelingdqn/target/test0304.pkl')
    
    

print('Complete')
logs.close()
plt.ioff()
plt.show()