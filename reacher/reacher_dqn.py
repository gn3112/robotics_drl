from pyrep_legacy import PyRep
import vrep
import os
from subprocess import Popen
from matplotlib import pyplot as plt
import time
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import torch.optim as optim
import random
import torch
from environment import environment
from PIL import Image
import math
import numpy as np

class Replay_Buffer(object):
# Transition: state, action, reward, next state
# List of of dim N x 4 with len() < capacity
# Dictionnary

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,transition):
        if self.__len__() == self.capacity:
            return 'full'
        else:
            if len(transition) != 4:
                raise RunTimeError('Your Transition is incomplete')
            self.memory.append(transition)
            self.position += 1 # Necessary ?
            return 'free'

    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self,h,w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16) # Batch norm for RL 50/50
        self.conv2 = nn.Conv2d(16,32,kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))

        self.fc1 = nn.Linear(conv_h*conv_w*32,2) # 2 discrete actions

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0),-1))

def select_actions(state,eps_start,eps_end,eps_decay,steps_done,policy_net):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) #check here
    else:
        return torch.tensor([[random.randrange(9)]], dtype=torch.long)

def evaluation():
    # Reward to achieve goals
    # Reward over training steps
    return None

def optimize_model(memory, gamma, batch_size):
    transitions = memory.sample(batch_size)

    state_batch = []
    action_batch = []
    reward_batch =  []
    state_next_batch = []

    for i in range(len(transitions)):
        transition = transitions[i]
        state_batch.append(transition["s"])
        action_batch.append(transition["a"])
        reward_batch.append(transition["r"])
        state_next_batch.append(transition["s'"])

    state_batch = torch.tensor(state_batch)
    action_batch = torch.tensor(action_batch)
    reward_batch =  torch.tensor(reward_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # next_state_values = torch.zeros(batch_size, device=device)
    next_state_values = target_net(state_next_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment()

    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        T.Resize(64, interpolation=Image.CUBIC),
                        T.ToTensor()])
    img = env.render()
    img = torch.from_numpy(img)
    img_height, img_width, _ = img.shape

    policy_net = DQN(img_height,img_width).to(device)
    target_net = DQN(img_height,img_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = Replay_Buffer(10000)

    epoch = 4
    batch_size = 128
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10

    num_episodes = 500

    # 1.Fill the replay buffer
    # Take action
    # Record s,a,r,s' in buffer
    # 2. Once full, how much to feed and how many forward pass with the
    # 3. Update policy network parameters
    # 4. Refill the buffer
    # 5. After some gradient step update target network

    obs = env.render()
    obs = resize(obs).unsqueeze(0).to(device)
    steps = 0
    steps_all = []
    # return_ = 0
    # return_all = []

    for _ in range(10):
        while True:
            action = select_actions(obs,
                                    eps_start,eps_end,eps_decay,steps,policy_net)
            action = action.to(device)
            reward = env.step_(action)
            obs_next = env.render()
            obs_next = resize(obs_next).unsqueeze(0).to(device)
            transition = {'s': obs,
                          'a': action,
                          'r': reward,
                          "s'": obs_next
                          }
            steps += 1
            # return_ += reward

            memory_state = memory.push(transition)

            obs = env.render()
            obs = resize(obs).unsqueeze(0).to(device)

            if memory_state == 'full':
                # Disregard last transition if incomplete or complete it
                break

            if reward == 1:
                # return_all.append(return_)
                # return_ = 0
                steps = 0
                steps_all.append(steps)
                print(steps_all)
                env.reset_target_position(random_=True)
                env.reset_robot_position(random_=False)
            elif steps == 3000:
                steps = 0
                env.reset_robot_position(random_=False)

        for _ in range(epoch):
            if epoch == 2: # update target network parameters
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
            for _ in range(memory.__len__()/batch_size):
                optimize_model(memory, gamma, batch_size)

if __name__ == "__main__":
    main()
