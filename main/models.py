import copy
import torch
from torch import nn
from torch.distributions import Distribution, Normal
from pyro.distributions import PlanarFlow, RadialFlow
from pyro.distributions.torch_transform import TransformModule
import torch.nn.functional as F
import os
import torchvision
from models_lpf import *

class TanhFlow(TransformModule):
  def __init__(self, size):
    super().__init__()

  def _call(self, x):
    return torch.tanh(x)

  def log_abs_det_jacobian(self, x, y):
    return torch.log1p(-y ** 2 + 1e-6).sum(dim=1)  # Uses log1p = log(1 + x) for extra numerical stability

# Normalising flow code from https://github.com/acids-ircam/pytorch_flows
class NormalisingFlow(nn.Module):
  def __init__(self, size, num_flows, flow_type='planar'):
    super().__init__()
    if flow_type == 'planar':
      flow = PlanarFlow
    elif flow_type == 'radial':
      flow = RadialFlow
    elif flow_type == 'tanh':
      flow = TanhFlow
    self.bijectors = nn.ModuleList([flow(size) for _ in range(num_flows)])

  def forward(self, x, log_p=None):
    for bijector in self.bijectors:
      y = bijector(x)
      log_p = None if log_p is None else log_p - bijector.log_abs_det_jacobian(x, y)  # Calculates log probability of value using the change-of-variables technique
      x = y
    return x, log_p

class Actor(nn.Module):
  def __init__(self, hidden_size, stochastic=True, layer_norm=False):
    super().__init__()
    layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.policy = nn.Sequential(*layers)
    if stochastic:
      self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))


  def forward(self, state):
    policy = self.policy(state)
    return policy

class TanhNormal(Distribution):
  def __init__(self, loc, scale):
    super().__init__()
    self.normal = Normal(loc, scale)

  def sample(self):
    return torch.tanh(self.normal.sample())

  def rsample(self):
    return torch.tanh(self.normal.rsample())

  def rsample_log_prob(self):
    value = self.normal.rsample()
    log_prob = self.normal.log_prob(value)
    value = torch.tanh(value)
    log_prob -= torch.log1p(-value.pow(2) + 1e-6)
    return value, log_prob.sum(dim=1)

  # Calculates log probability of value using the change-of-variables technique (uses log1p = log(1 + x) for extra numerical stability)
  def log_prob(self, value):
    inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
    return self.normal.log_prob(inv_value).view(-1,1) - torch.log1p(-value.pow(2) + 1e-6).view(-1,1)  # log p(f^-1(y)) + log |det(J(f^-1(y)))|

  @property
  def mean(self):
    return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):
  def __init__(self, hidden_size, action_space, obs_space, flow_type='radial', flows=0):
    super().__init__()
    self.hidden_size = hidden_size
    self.action_space = action_space
    self.obs_space = obs_space
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies

    layers = [nn.Linear(self.obs_space[0], hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, self.action_space*2)] # nn.Softmax(dim=0))
    self.policy = nn.Sequential(*layers)

    flows, flow_type = (1, 'tanh') if flows == 0 else (flows, flow_type)
    self.flows = NormalisingFlow(self.action_space, flows, flow_type=flow_type)

  def forward(self, state, log_prob=False, deterministic=False):
    policy_mean, policy_log_std = self.policy(state).view(-1,self.action_space*2).chunk(2,dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    base_distribution = Normal(policy_mean, policy_log_std.exp())
    action = base_distribution.mean if deterministic else base_distribution.rsample()
    log_p = base_distribution.log_prob(action).sum(dim=1) if log_prob else None
    action, log_p = self.flows(action, log_p)
    return action, log_p

class SoftActorConv(nn.Module):
  def __init__(self, hidden_size, action_space, flow_type='radial', flows=0):
    super().__init__()
    self.hidden_size = hidden_size
    self.action_space = action_space
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    self.conv1 = nn.Conv2d(4,16,kernel_size=4, stride=3) # Check here for dim (frame staking)
    self.conv2 = nn.Conv2d(16,32,kernel_size=4, stride=3)
    self.conv3 = nn.Conv2d(32,32,kernel_size=4, stride=3)

    def conv2d_size_out(size, kernel_size = 4, stride = 3):
        return (size - (kernel_size - 1) - 1) // stride + 1

    conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
    conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))

    self.fc1 = nn.Linear(conv_h*conv_w*32,hidden_size)
    self.fc2 = nn.Linear(hidden_size,self.action_space*2)

    self.flows = NormalisingFlow(self.action_space, flows, flow_type=flow_type)

  def forward(self, state, log_prob=False, deterministic=False):
    #torchvision.utils.save_image(state.view(4,64,64)[0,:,:],"sac_image_network.png",normalize=True)
    x = F.relu((self.conv1(state.view(-1,4,64,64))))
    x = F.relu((self.conv2(x)))
    x = F.relu((self.conv3(x)))
    x = F.relu(self.fc1(x.view(x.size(0),-1)))
    x = (self.fc2(x))
    policy_mean, policy_log_std = x.view(-1,self.action_space*2).chunk(2,dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    base_distribution = Normal(policy_mean, policy_log_std.exp())
    action = base_distribution.mean if deterministic else base_distribution.rsample()
    log_p = base_distribution.log_prob(action).sum(dim=1) if log_prob else None
    action, log_p = self.flows(action, log_p)
    return action, log_p

class SoftActorGated(nn.Module):
  def __init__(self, hidden_size, action_space, obs_space, flow_type='radial', flows=0):
    super().__init__()
    self.hidden_size = hidden_size
    self.action_space = action_space
    self.obs_space = obs_space
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies

    self.fc1 = nn.Linear(self.obs_space[0], hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.fc4 = nn.Linear(hidden_size, self.action_space * 2)

    flows, flow_type = (1, 'tanh') if flows == 0 else (flows, flow_type)
    self.flows = NormalisingFlow(self.action_space, flows, flow_type=flow_type)

  def forward(self, state, log_prob=False, deterministic=False):
    x = F.tanh(self.fc1(state))
    x = F.tanh(self.fc2(x))
    gate = F.sigmoid(self.fc3(x)).view(-1,1) #Try 1-p with 1
    device = gate.get_device()
    neg_gate = torch.tensor([1.]).view(-1,1).repeat(gate.size()[0], 1).to(device) - gate
    policy = self.fc4(x).view(-1,self.action_space*2)
    policy[:,:3] = torch.matmul(policy[:,:3].clone().view(-1,3,1), gate.view(-1,1,1)).view(-1,3)
    policy[:,3:8] = torch.matmul(policy[:,3:8].clone().view(-1,5,1), neg_gate.view(-1,1,1)).view(-1,5)
    policy[:,8:11] = torch.matmul(policy[:,8:11].clone().view(-1,3,1), gate.view(-1,1,1)).view(-1,3)
    policy[:,11:] = torch.matmul(policy[:,11:].clone().view(-1,5,1), neg_gate.view(-1,1,1)).view(-1,5)

    policy_mean, policy_log_std = policy.chunk(2,dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    base_distribution = Normal(policy_mean, policy_log_std.exp())
    action = base_distribution.mean if deterministic else base_distribution.rsample()
    log_p = base_distribution.log_prob(action).sum(dim=1) if log_prob else None
    action, log_p = self.flows(action, log_p)
    return action, log_p, gate

class ActorImageNet(nn.Module):
    def __init__(self, hidden_size, action_space, obs_space, flow_type='radial', flows=0):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
        self.obs_space = obs_space[0] - 5
        self.action_space = action_space
        self.conv1 = nn.Conv2d(12,48,kernel_size=6, stride=2) # Check here for dim (frame staking)
        # self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down1 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        self.conv2 = nn.Conv2d(48, 64,kernel_size=4, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down2 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5

        self.fc1 = nn.Linear(self.obs_space, 256)
        # self.fc2 = nn.Linear(256,64)

        # self.conv3 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.down3 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        # self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.fc3 = nn.Linear(36864,self.action_space*2)
        self.fc2 = nn.Linear(223040,256)
        self.fc3 = nn.Linear(256,256)
        self.fc_gate = nn.Linear(256, 1)
        self.fc4 = nn.Linear(256,self.action_space*2)

        flows, flow_type = (1, 'tanh') if flows == 0 else (flows, flow_type)
        self.flows = NormalisingFlow(self.action_space, flows, flow_type=flow_type)

    def forward(self, obs, log_prob=False, deterministic=False):
        high_obs = obs['high'].view(-1,12,128,128)
        low_obs = obs['low'].view(-1,32)[:,:27]
        high_x = F.relu(self.conv1(high_obs))
        high_x = F.relu(self.conv2(high_x))
        print(torch.mean(high_x))
        print(torch.mean(self.conv2.weight.data))
        low_x = F.relu(self.fc1(low_obs)).view(-1,256)
        # low_x = F.relu(self.fc2(low_x)).view(-1,64,1,1)
        #
        # x = F.relu(self.conv3(high_x + low_x))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.fc2(torch.cat((high_x.view(high_x.size(0),-1), low_x),dim=1)))
        x = F.relu(self.fc3(x))
        gate = F.sigmoid(self.fc_gate(x)).view(-1,1)
        device = gate.get_device()
        neg_gate = torch.tensor([1.]).view(-1,1).repeat(gate.size()[0], 1).to(device) - gate
        policy = self.fc4(x).view(-1,self.action_space*2)
        policy[:,:3] = torch.matmul(policy[:,:3].clone().view(-1,3,1), gate.view(-1,1,1)).view(-1,3)
        policy[:,3:8] = torch.matmul(policy[:,3:8].clone().view(-1,5,1), neg_gate.view(-1,1,1)).view(-1,5)
        policy[:,8:11] = torch.matmul(policy[:,8:11].clone().view(-1,3,1), gate.view(-1,1,1)).view(-1,3)
        policy[:,11:] = torch.matmul(policy[:,11:].clone().view(-1,5,1), neg_gate.view(-1,1,1)).view(-1,5)

        policy_mean, policy_log_std = policy.view(-1,self.action_space*2).chunk(2,dim=1)
        policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
        base_distribution = Normal(policy_mean, policy_log_std.exp())
        action = base_distribution.mean if deterministic else base_distribution.rsample()
        log_p = base_distribution.log_prob(action).sum(dim=1) if log_prob else None
        action, log_p = self.flows(action, log_p)
        print(action)
        return action, log_p

class CriticImageNet(nn.Module):
    def __init__(self, hidden_size, output_size, obs_space, action_space, state_action=False, layer_norm=False):
        super().__init__()
        self.obs_space = obs_space[0]
        self.action_space = action_space

        # self.conv1 = nn.Conv2d(9,64,kernel_size=6, stride=2) # Check here for dim (frame staking)
        # self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down1 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        # self.conv2 = nn.Conv2d(64, 64,kernel_size=5, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down2 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        #
        # self.fc1 = nn.Linear(self.obs_space + self.action_space, 256)
        # self.fc2 = nn.Linear(256,64)
        #
        # self.conv3 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.down3 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        # self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.fc3 = nn.Linear(36864, output_size)

        self.conv1 = nn.Conv2d(9,32,kernel_size=6, stride=2) # Check here for dim (frame staking)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down1 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        self.conv2 = nn.Conv2d(32, 32,kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.down2 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5

        self.fc1 = nn.Linear(self.obs_space + self.action_space, 256)
        # self.fc2 = nn.Linear(256,64)

        # self.conv3 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.down3 = Downsample(channels=64, filt_size=3, stride=2) #filt_size is blur kernel 3 or 5
        # self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.fc3 = nn.Linear(36864,self.action_space*2)
        self.fc2 = nn.Linear(93568,256)
        self.fc3 = nn.Linear(256,output_size)

    def forward(self, obs, action=None):
        high_obs = obs['high'].view(-1,9,128,128)
        low_obs = obs['low'].view(-1,27)
        # high_x = F.relu(self.conv1(high_obs))
        # high_x = F.relu(self.conv2(self.down1(self.max_pool1(high_x))))
        # high_x = self.down2(self.max_pool2(high_x))
        #
        # low_x = F.relu(self.fc1(torch.cat((low_obs, action.view(-1,8)),dim=1)))
        # low_x = F.relu(self.fc2(low_x)).view(-1,64,1,1)
        #
        # x = F.relu(self.conv3(high_x + low_x))
        # x = F.relu(self.conv4(x))
        # x = self.fc3(x.view(x.size(0),-1))
        high_x = F.relu(self.conv1(high_obs))
        high_x = F.relu(self.conv2(self.max_pool1(high_x)))
        high_x = (self.max_pool2(high_x))

        low_x = F.relu(self.fc1(torch.cat((low_obs, action.view(-1,8)),dim=1))).view(-1,256)
        # low_x = F.relu(self.fc2(low_x)).view(-1,64,1,1)
        #
        # x = F.relu(self.conv3(high_x + low_x))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.fc2(torch.cat((high_x.view(high_x.size(0),-1), low_x),dim=1)))
        x = self.fc3(x)

        return x.squeeze(dim=1)

class SoftActorFork(nn.Module):
  def __init__(self, hidden_size, action_space, obs_space, flow_type='radial', flows=0):
    super().__init__()
    self.hidden_size = hidden_size
    self.action_space = action_space
    self.obs_space = obs_space
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies

    self.fc1 = nn.Linear(self.obs_space[0], hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

    self.fc4_1 = nn.Linear(hidden_size, hidden_size)
    self.fc5_1 = nn.Linear(hidden_size, 3 * 2)
    self.fc4_2 = nn.Linear(hidden_size, hidden_size)
    self.fc5_2 = nn.Linear(hidden_size, 5 * 2)

    flows, flow_type = (1, 'tanh') if flows == 0 else (flows, flow_type)

    self.flows_base = NormalisingFlow(3, flows, flow_type=flow_type)
    self.flows_arm = NormalisingFlow(5, flows, flow_type=flow_type)

  def forward(self, state, log_prob=False, deterministic=False):
    x = F.tanh(self.fc1(state))
    x = F.tanh(self.fc2(x))
    gate = F.sigmoid(self.fc3(x)).view(-1,1)
    device = gate.get_device()
    neg_gate = torch.tensor([1.]).view(-1,1).repeat(gate.size()[0], 1).to(device) - gate
    x_base = torch.matmul(x.view(-1,256,1), gate.view(-1,1,1)).view(-1,256)
    x_arm = torch.matmul(x.view(-1,256,1), gate.view(-1,1,1)).view(-1,256)
    x_base = F.tanh(self.fc4_1(x_base))
    x_arm = F.tanh(self.fc4_2(x_arm))

    x_base = self.fc5_1(x_base)
    x_arm = self.fc5_2(x_arm)

    policy_mean_base, policy_log_std_base = x_base.view(-1,(3 * 2)).chunk(2,dim=1)
    policy_mean_arm, policy_log_std_arm = x_arm.view(-1,(5 * 2)).chunk(2,dim=1)

    policy_log_std_base = torch.clamp(policy_log_std_base, min=self.log_std_min, max=self.log_std_max)
    policy_log_std_arm = torch.clamp(policy_log_std_arm, min=self.log_std_min, max=self.log_std_max)

    base_distribution_base = Normal(policy_mean_base, policy_log_std_base.exp())
    base_distribution_arm = Normal(policy_mean_arm, policy_log_std_arm.exp())

    action_base = base_distribution_base.mean if deterministic else base_distribution_base.rsample()
    action_arm = base_distribution_arm.mean if deterministic else base_distribution_arm.rsample()

    log_p_base = base_distribution_base.log_prob(action_base).sum(dim=1) if log_prob else None
    log_p_arm = base_distribution_arm.log_prob(action_arm).sum(dim=1) if log_prob else None

    action_base, log_p_base = self.flows_base(action_base, log_p_base)
    action_arm, log_p_arm = self.flows_arm(action_arm, log_p_arm)

    action = torch.cat((action_base,action_arm), dim=1)

    if log_prob:
        log_p = (log_p_base + log_p_arm)/2
    else:
        log_p = None
    return action, log_p

class Critic(nn.Module):
  def __init__(self, hidden_size, output_size, obs_space, action_space, state_action=False, layer_norm=False):
    super().__init__()
    self.state_action = state_action
    self.action_space = action_space
    self.obs_space = obs_space

    if len(self.obs_space) > 1:
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=3) # Check here for dim (frame staking)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(32,32,kernel_size=4, stride=3)

        def conv2d_size_out(size, kernel_size = 4, stride = 3):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
        self.fc1 = nn.Linear(conv_h*conv_w*32 + (self.action_space if self.state_action else 0), hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    else:
        layers = [nn.Linear(self.obs_space[0] + (self.action_space if self.state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, output_size)]
        if layer_norm:
          layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
        self.value = nn.Sequential(*layers)

  def forward(self, state, action=None):
    if len(self.obs_space) > 1:
        x = F.relu((self.conv1(state.view(-1,4,64,64))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        if self.state_action:
            x = F.relu(self.fc1(torch.cat([x.view(x.size(0),-1), action], dim=1)))
        else:
            x = F.relu(self.fc1(x.view(x.size(0),-1)))

        value = (self.fc2(x))

    else:
        if self.state_action:
            value = self.value(torch.cat([state.view(-1,self.obs_space[0]), action.view(-1,self.action_space)], dim=1))
        else:
            value = self.value(state)

    return value.squeeze(dim=1)

class ActorCritic(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.actor = Actor(hidden_size, stochastic=True)
    self.critic = Critic(hidden_size)

  def forward(self, state):
    policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
    value = self.critic(state)
    return policy, value

class DQN_FC(nn.Module):

    def __init__(self):
        super(DQN_FC, self).__init__()
        self.fc1 = nn.Linear(10,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 8)

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

def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network

def update_target_network(network, target_network, polyak_factor):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data
