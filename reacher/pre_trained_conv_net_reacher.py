from collections import deque
from env_reacher_v2 import environment
import copy
import torch
from torch import optim
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import logz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4,16,kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32,32,kernel_size=4, stride=2)

        def conv2d_size_out(size, kernel_size = 4, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))

        self.fc1 = nn.Linear(conv_h*conv_w*32,512)
        self.fc2 = nn.Linear(512,6)

    def forward(self, state):
        x = F.relu((self.conv1(state.view(-1,4,64,64))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        print(x)
        x = F.relu(self.fc1(x.view(x.size(0),-1)))
        x = (self.fc2(x))
        joint_pos, target_pos, joint_vel = x.view(-1,6).chunk(3,dim=1)
        return joint_pos, target_pos, joint_vel

def to_torch(a):
    return torch.tensor(a, dtype=torch.float32, device=device)

def get_loss(D,BATCH_SIZE,net):
    batch = random.sample(D, BATCH_SIZE)
    img_batch =  []
    target_pos_batch = []
    joint_pos_batch = []
    joint_vel_batch = []
    for d in batch:
        target_pos_batch.append(d['target_pos'])
        joint_pos_batch.append(d['joint_pos'])
        joint_vel_batch.append(d['joint_vel'])
        img_batch.append(d['img'])
            
        batch = {'target_pos':torch.cat(target_pos_batch,dim=0),
                 'joint_pos':torch.cat(joint_pos_batch,dim=0),
                 'joint_vel':torch.cat(joint_vel_batch,dim=0),
                 'img':torch.cat(img_batch,dim=0)
                }
        
    pred_joint_pos, pred_target_pos, pred_joint_vel = net(torch.randn(64,4,64,64).float().to(device))
    loss = ((batch['target_pos'] - pred_target_pos).pow(2).sum(dim=1) + (batch['joint_pos'] - pred_joint_pos).pow(2).sum(dim=1) + (batch['joint_vel'] - pred_joint_vel).pow(2).sum(dim=1)).mean()
    return loss

def main():
    DATASET_SIZE = 200
    STEPS = 3000
    VALIDSET_SIZE = 64
    LR = 0.7
    BATCH_SIZE = 64

    if not(os.path.exists('data/pre_trained_model')):
                os.makedirs('data/pre_trained_model')

    home = os.path.expanduser('~')
    expdir = os.path.join(home,'robotics_drl/reacher/data/pre_trained_model')
    logz.configure_output_dir(d=expdir)
    
    D = deque(maxlen=DATASET_SIZE)
    V = deque(maxlen=VALIDSET_SIZE)
    env = environment(continuous_control=True, obs_lowdim=False, rpa=4, frames=4)
    env.reset()
    net = network().to(device)
    optimiser = optim.Adam(net.parameters(), lr=LR)

    pbar = tqdm(range(1, STEPS + 1), unit_scale=1, smoothing=0)

    for _ in range(DATASET_SIZE):
        action = env.sample_action()
        obs, _, _ = env.step(action)
        target_pos = env.target_position()
        joint_pos = env.agent.get_joint_positions()
        joint_vel = env.agent.get_joint_velocities()
        
        D.append({"target_pos": to_torch(target_pos[:2]).view(1,-1), "joint_pos": to_torch(joint_pos).view(1,-1), "joint_vel": to_torch(joint_vel).view(1,-1), "img": to_torch(obs).unsqueeze(dim=0)})

    for _ in range(VALIDSET_SIZE):
        action = env.sample_action()
        obs, _, _ = env.step(action)
        target_pos = env.target_position()
        joint_pos = env.agent.get_joint_positions()
        joint_vel = env.agent.get_joint_velocities()

        V.append({"target_pos": to_torch(target_pos[:2]).view(1,-1), "joint_pos": to_torch(joint_pos).view(1,-1), "joint_vel": to_torch(joint_vel).view(1,-1), "img": to_torch(obs).unsqueeze(dim=0)})


    for step in pbar:
        if len(D) > BATCH_SIZE * 2:
            loss = get_loss(D,BATCH_SIZE,net)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            if step % 600 == 0 and step != 0:
                net.eval()
                loss_v = get_loss(V,VALIDSET_SIZE,net)
                pbar.set_description('Loss training: %s | Loss validation: %s' %(loss.item(), loss_v.item())) 
                net.train()
                logz.log_tabular('Loss training',loss.item())
                logz.log_tabular('Loss validation',loss_v.item())
                logz.dump_tabular()

    home = os.path.expanduser("~")
    path = home + "/robotics_drl/reacher/pre_trained_net_reacher"
    torch.save(net.state_dict(), os.path.join(path, "model.pkl"))
    env.terminate()

if __name__ == "__main__":
    main()
