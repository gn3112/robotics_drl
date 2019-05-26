import os
from subprocess import Popen
from matplotlib import pyplot as plt
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import torch.optim as optim
import random
from PIL import Image
import math
import numpy as np
import logz
import inspect
from collections import deque
from environment import environment
from images_to_video import im_to_vid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Replay_Buffer(object):
# Transition: state, action, reward, next state
# List of of dim N x 4 with len() < capacity
# Dictionnary

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=self.capacity)

    def push(self,transition):
        if self.__len__() == self.capacity:
            return 'full'
        else:
            if len(transition) != 4:
                raise RunTimeError('Your Transition is incomplete')

            self.memory.appendleft(transition)
            return 'free'

    def clear_(self):
        self.memory.clear() # len() = 0

    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_FC(nn.Module):

    def __init__(self):
        super(DQN_FC, self).__init__()
        self.fc1 = nn.Linear(10,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu((self.fc1(x.float())))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return x

class DQN(nn.Module):

    def __init__(self,h,w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16) # Batch norm for RL 50/50
        self.conv2 = nn.Conv2d(16,32,kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))

        self.fc1 = nn.Linear(conv_h*conv_w*32,8) # dont hardcode num of actions

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.fc1(x.view(x.size(0),-1))

class evaluation(object):
    def __init__(self, env, expdir, n_states=20):
        self.states_eval = []
        self.env = env
        self.imtovid = im_to_vid(expdir)
        self.expdir = expdir
        self.ep = 0
        # self.resize = T.Compose([T.ToPILImage(),
        #                          T.Grayscale(num_output_channels=1),
        #                          T.Resize(64, interpolation=Image.BILINEAR),
        #                          T.ToTensor()])
        for _ in range(n_states):
            self.env.reset_robot_position(random_=True)
            self.env.reset_target_position(random_=True)
            obs = ((self.env.get_obs()))
            self.states_eval.append(obs)

    def get_qvalue(self,policy_net):
        policy_net.eval()
        qvalues = 0
        for _, obs in enumerate(self.states_eval):
            with torch.no_grad():
                qvalues += policy_net(torch.from_numpy(obs).view(1,-1).to(device)).max(1)[0]
        return (qvalues/len(self.states_eval))[0].item()

    def sample_episode(self,policy_net,save_video=False,n_episodes=5,threshold_ep=60):
        # 0.1 greedy policy or 100% action from network ?
        policy_net.eval()
        steps_all = []
        return_all = []
        for _ in range(n_episodes):
            steps = 0
            return_ = 0
            img_ep = deque([])
            self.env.reset_robot_position(random_=True)
            self.env.reset_target_position(random_=True)
            while True:
                obs = (self.env.get_obs())
                img = self.env.render()
                img_ep.append(img)
                with torch.no_grad():
                    action = policy_net(torch.from_numpy(obs).view(1,-1).to(device)).argmax(1).view(1,-1)
                reward, done = self.env.step_(action)
                steps += 1
                return_ += reward
                if done:
                    break
                elif steps == threshold_ep:
                    break

            if save_video==True: self.save_ep_video(img_ep)
            steps_all.append(steps)
            return_all.append(return_)
        return return_all, steps_all

    def save_ep_video(self,imgs):
        self.ep += 1
        self.imtovid.from_list(imgs,self.ep)

    def record_episode(self,img_all):
        logdir = os.path.expanduser('~') + '/robotics_drl/reacher/' + self.expdir + '/episode%s'%(self.ep)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        size = 64, 64
        for idx, img in enumerate(img_all):
            ndarr = img.reshape(64,64,3)
            im = Image.fromarray(ndarr)
            imdir = os.path.join(logdir,'step%s.jpg'%idx)
            im.resize(size, Image.BILINEAR)
            im.save(imdir,"JPEG")
        self.ep += 1

def select_actions(state,eps_start,eps_end,eps_decay,steps_done,policy_net,env):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        policy_net.eval()
        return (policy_net((state.to(device))).argmax(1).view(1,-1)), eps_threshold
    else:
        return torch.tensor([[random.randrange(len(env.action_all))]], dtype=torch.long), eps_threshold

def optimize_model(policy_net,target_net, optimizer, memory, gamma, batch_size):
    if memory.__len__() < batch_size*50:
        return False

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
    state_action_values = policy_net((state_batch)).gather(1, action_batch)
    for param in target_net.parameters():
        param.requires_grad = False

    next_state_values = torch.zeros((batch_size),device=device)
    next_state_values[non_final_idx] = target_net(state_next_batch_nf.to(device)).max(1)[0]
    expected_state_action_values = (next_state_values.view(-1,1) * gamma) + reward_batch.to(device)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(episodes, learning_rate, batch_size, gamma, eps_start, eps_end,
          eps_decay, target_update, max_steps, buffer_size,
          random_link, random_target, repeat_actions, logdir):

    setup_logger(logdir, locals())

    env = environment()

    eval_policy = evaluation(env,logdir)

    env.reset_target_position(random_=True)
    env.reset_robot_position(random_=False)

    # resize = T.Compose([T.ToPILImage(),
    #                     T.Grayscale(num_output_channels=1),
    #                     T.Resize(64, interpolation = Image.BILINEAR),
    #                     T.ToTensor()])
    img = env.get_obs()
    img = torch.from_numpy(img.copy())
    # img_height, img_width, _ = img.shape

    policy_net = DQN_FC().to(device)
    target_net = DQN_FC().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)
    memory = Replay_Buffer(buffer_size)

    obs = env.get_obs()
    obs = torch.from_numpy((obs)).view(1,-1)

    successes = 0

    target_upd = 0
    grad_upd = 0
    steps_train = 0

    for ep in range(1,episodes+1):
        env.reset_robot_position(random_=random_link)
        env.reset_target_position(random_=random_target) #target after link reset so vel=0
        rewards_ep = 0
        steps_ep = 0
        steps_all = []
        rewards_all = []
        sampling_time = 0
        start_time = time.time()
        while True:
            action, eps_threshold = select_actions(obs, eps_start, eps_end,
                                                       eps_decay, steps_train, policy_net,env)

            reward, done = env.step_(action)
            reward = torch.tensor(reward,dtype=torch.float).view(-1,1)
            obs_next = env.get_obs()
            obs_next = torch.from_numpy(obs_next).view(1,-1).to(device)
            transition = {'s': obs.to(device),
                          'a': action.to(device),
                          'r': reward,
                          "s'": obs_next.to(device)
                          }
            steps_ep += 1
            steps_train += 1
            rewards_ep += reward

            memory_state = memory.push(transition)

            obs = env.get_obs()
            obs = torch.from_numpy((obs)).view(1,-1)

            if done:
                rewards_all.append(rewards_ep/steps_ep)
                steps_all.append(steps_ep)
                successes += 1
                break

            elif steps_ep == max_steps:
                rewards_all.append(rewards_ep)
                steps_all.append(steps_ep)
                break

            status = optimize_model(policy_net, target_net, optimizer, memory, gamma, batch_size)
            if status != False:
                grad_upd += 1
                if grad_upd % target_update == 0:# update target network parameters
                    target_net.load_state_dict(policy_net.state_dict())
                    target_net.eval()
                    target_upd += 1



        end_time = time.time()
        sampling_time += end_time-start_time
        sampling_time /= ep

        if ep % 20 == 0:
            return_val, steps_val = eval_policy.sample_episode(policy_net,save_video=True if ep%500==0 else False, n_episodes=5)
            qvalue_eval = eval_policy.get_qvalue(policy_net)
            logz.log_tabular('Averaged Steps Traning',np.around(np.average(steps_all),decimals=0)) # last 10 episodes
            logz.log_tabular('Averaged Return Training',np.around(np.average(rewards_all),decimals=2))
            logz.log_tabular('Averaged Steps Validation',np.around(np.average(steps_val),decimals=0))
            logz.log_tabular('Averaged Return Validation',np.around(np.average(return_val),decimals=2))
            logz.log_tabular('Cumulative Successes',successes)
            logz.log_tabular('Number of episodes',ep)
            logz.log_tabular('Sampling time (s)', sampling_time)
            logz.log_tabular('Epsilon threshold', eps_threshold)
            logz.log_tabular('Gradient update', grad_upd )
            logz.log_tabular('Updates target network', target_upd)
            logz.log_tabular('Average q-value evaluation', qvalue_eval)
            logz.dump_tabular()
            steps_all = []
            rewards_all = []
            logz.save_pytorch_model(policy_net.state_dict())

    env.terminate()

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
    parser.add_argument('-lr','--learning_rate',default=0.0002,type=float)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('-bs','--batch_size',default=64,type=int)
    parser.add_argument('-buffer_size',default=100000,type=int)
    parser.add_argument('-ep','--episodes',default=15000,type=int)
    parser.add_argument('-target_update',default=7500,type=int,help='every n gradient steps')
    parser.add_argument('-max_steps',default=150,type=int)
    parser.add_argument('-gamma',default=0.999,type=float)
    parser.add_argument('-eps_start',default=0.9,type=float)
    parser.add_argument('-eps_end',default=0.05,type=float)
    parser.add_argument('-eps_decay',default=80000,type=float)
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

    start = time.time()

    train(args.episodes,
          args.learning_rate,
          args.batch_size,
          args.gamma,
          args.eps_start,
          args.eps_end,
          args.eps_decay,
          args.target_update,
          args.max_steps,
          args.buffer_size,
          args.random_link,
          args.random_target,
          args.repeat_actions,
          logdir)

    end = time.time()

    print("Total running time %s"%str(end-start))
if __name__ == "__main__":
    main()
