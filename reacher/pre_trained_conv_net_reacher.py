from collections import deque
from env_reacher_v2 import environment
import copy
import torch
from torch import optim
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F
from tqdm import tqdm

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

        self.fc1 = nn.Linear(conv_h*conv_w*32,256)
        self.fc2 = nn.Linear(256,6)

    def forward(self, state):
        x = F.relu((self.conv1(state.view(-1,4,64,64))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0),-1)))
        x = F.tanh(self.fc2(x))
        joint_pos, target_pos, joint_vel = x.view(-1,6).chunk(3,dim=1)
        return joint_pos, target_pos, joint_vel

def to_torch(a):
    return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(dim=0)

def main():
    DATASET_SIZE = 50000
    STEPS = 100000
    LR = 0.0001
    BATCH_SIZE = 64

    D = deque(maxlen=DATASET_SIZE)

    env = environment(continuous_control=True, obs_lowdim=False, rpa=3, frames=1)
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

        D.append({"target_pos": to_torch(target_pos), "joint_pos": to_torch(joint_pos), "joint_vel": to_torch(joint_vel), "img": to_torch(obs)})


    for step in pbar:
        if len(D) > BATCH_SIZE * 10:
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
            pred_joint_pos, pred_target_pos, pred_joint_vel = net(batch["img"])
            loss = ((batch['target_pos'] - pred_target_pos).pow(2).sum(dim=1) + (batch['joint_pos'] - pred_joint_pos).pow(2).sum(dim=1) + (batch['joint_vel'] - pred_joint_vel).pow(2).sum(dim=1)).mean()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            pbar.set_description('Loss: %i' %(loss))

    home = expanduser("~")
    path = home + "/robotics_drl/reacher/pre_trained_net_reacher"
    torch.save(model, osp.join(path, "model.pkl"))


if __name__ == "__main__":
    main()
