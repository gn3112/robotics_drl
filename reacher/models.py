import copy
import torch
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F

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
  def __init__(self, hidden_size, action_space, obs_space):
    super().__init__()
    self.action_space = action_space
    self.obs_space = obs_space
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    if len(self.obs_space) > 1:
        self.conv1 = nn.Conv2d(4,16,kernel_size=4, stride=2) # Check here for dim (frame staking)
        self.conv2 = nn.Conv2d(16,32,kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32,32,kernel_size=4, stride=2)

        def conv2d_size_out(size, kernel_size = 4, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(64)))

        self.fc1 = nn.Linear(conv_h*conv_w*32,hidden_size)
        self.fc2 = nn.Linear(hidden_size,self.action_space*2)
    else:
        layers = [nn.Linear(self.obs_space[0], hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, self.action_space*2)] # nn.Softmax(dim=0))
        self.policy = nn.Sequential(*layers)

  def forward(self, state):
    if len(self.obs_space) > 1:
        x = F.relu((self.conv1(state.view(-1,4,64,64))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0),-1)))
        x = F.relu(self.fc2(x))
        policy_mean, policy_log_std = x.view(-1,self.action_space*2).chunk(2,dim=1)
    else:
        policy_mean, policy_log_std = self.policy(state).view(-1,self.action_space*2).chunk(2,dim=1)

    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    policy = TanhNormal(policy_mean, policy_log_std.exp())

    return policy

class Critic(nn.Module):
  def __init__(self, hidden_size, output_size, obs_space, action_space, state_action=False, layer_norm=False):
    super().__init__()
    self.state_action = state_action
    self.action_space = action_space
    self.obs_space = obs_space

    if len(self.obs_space) > 1:
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2) # Check here for dim (frame staking)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32,32,kernel_size=4, stride=2)

        def conv2d_size_out(size, kernel_size = 4, stride = 2):
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

        value = F.relu(self.fc2(x))

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
