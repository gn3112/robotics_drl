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
import logz
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Replay_Buffer(object):
#python maxlem buffer
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

        self.fc1 = nn.Linear(conv_h*conv_w*32,9)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0),-1))

def select_actions(state,eps_start,eps_end,eps_decay,steps_done,policy_net):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        policy_net.eval()
        return policy_net(state).argmax(1).view(1,-1), eps_threshold
    else:
        return torch.tensor([[random.randrange(9)]], dtype=torch.long), eps_threshold

def evaluation():
    # Reward to achieve goals
    # Reward over training steps
    return None

def optimize_model(policy_net,target_net, optimizer, memory, gamma, batch_size, device):
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

    state_batch = torch.cat(state_batch,dim=0)
    action_batch = torch.cat(action_batch,dim=0)
    reward_batch =  torch.cat(reward_batch,dim=0)
    state_next_batch = torch.cat(state_next_batch,dim=0)

    non_final_idx = []
    for idx, r in enumerate(reward_batch):
        if r != 1:
            non_final_idx.append(idx)

    non_final_idx = torch.tensor(non_final_idx,dtype=torch.long)
    state_next_batch_nf = state_next_batch[non_final_idx]

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    for param in target_net.parameters():
        param.requires_grad = False

    next_state_values = torch.zeros(batch_size,device=device)
    next_state_values[non_final_idx] = target_net(state_next_batch_nf).max(1)[0]
    expected_state_action_values = (next_state_values * gamma) + reward_batch.to(device)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(epoch, learning_rate, batch_size, gamma, eps_start, eps_end,
          eps_decay, policy_update, target_update, max_steps, buffer_size,
          random_link, random_target, repeat_actions, logdir):

    setup_logger(logdir, locals())


    env = environment()
    env.reset_target_position(random_=True)
    env.reset_robot_position(random_=False)

    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        T.Resize(64, interpolation=Image.BILINEAR),
                        T.ToTensor()])
    img = env.render()
    img = torch.from_numpy(img.copy())
    img_height, img_width, _ = img.shape

    policy_net = DQN(img_height,img_width).to(device)
    target_net = DQN(img_height,img_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr = learning_rate)
    memory = Replay_Buffer(buffer_size)

    obs = env.render()
    obs = resize(obs).unsqueeze(0).to(device)
    steps = 0
    ep = 0
    steps_all = []
    reward_avg = 0
    for update in range(policy_update):
        while True: # Sample transitions
            if ep > 21 or (ep < 20 and steps%4 == 0):
                action, eps_threshold = select_actions(obs,eps_start,eps_end,eps_decay,steps,policy_net)
                action = action.to(device)

            reward = env.step_(action)
            reward = torch.tensor(reward,dtype=torch.float).view(-1,1)
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
                # TODO: Disregard last transition if incomplete or complete it
                break

            if reward == 100:
                # return_all.append(return_)
                # return_ = 0
                ep += 1
                steps_all.append(steps-sum(steps_all))
                env.reset_target_position(random_=random_target)
                env.reset_robot_position(random_=random_link)
            elif steps == 1500:
                steps_all.append(steps-sum(steps_all))
                env.reset_target_position(random_=False)
                env.reset_robot_position(random_=False)

            reward_avg += reward
            reward_avg /= steps
            # if steps%50 == 0:
            #     print('--- Epsilon threshold: ' + str(eps_threshold) + ' Averaged Return: ' + str(reward_avg) + ' ---')

        logz.log_tabular('Cumulative Averaged Steps',np.average(steps_all))
        logz.log_tabular('Cumulative Averaged Returns',reward_avg)
        logz.log_tabular('Update',update)
        logz.log_tabular('Number of episodes',len(steps_all))
        logz.dump_tabular()

        for _ in range(epoch):
            if epoch == 4: # update target network parameters
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
            for _ in range(memory.__len__()//batch_size):
                optimize_model(policy_net, target_net, optimizer, memory, gamma, batch_size,device)

        memory = Replay_Buffer(buffer_size)
        logz.save_pytorch_model(policy_net.state_dict())

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(params)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--learning_rate',default=0.0001,type=float)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('-bs','--batch_size',default=128,type=int)
    parser.add_argument('-buffer_size',default=30000,type=int)
    parser.add_argument('-ep','--epoch',default=8,type=int)
    parser.add_argument('-policy_update',default=12,type=int)
    parser.add_argument('-target_update',default=160,type=int,help='gradient step')
    parser.add_argument('-max_steps',default=1200,type=int)
    parser.add_argument('-gamma',default=0.9999,type=float)
    parser.add_argument('-eps_start',default=0.9,type=float)
    parser.add_argument('-eps_end',default=0.1,type=float)
    parser.add_argument('-eps_decay',default=10000,type=float)
    parser.add_argument('-randL','--random_link',action='store_true')
    parser.add_argument('-randT','--random_target',action='store_true')
    parser.add_argument('-rpa','--repeat_actions',action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    train(args.epoch,
          args.learning_rate,
          args.batch_size,
          args.gamma,
          args.eps_start,
          args.eps_end,
          args.eps_decay,
          args.policy_update,
          args.target_update,
          args.max_steps,
          args.buffer_size,
          args.random_link,
          args.random_target,
          args.repeat_actions,
          logdir)


if __name__ == "__main__":
    main()
