from env_reacher_v2 import environment
import torch


class Env():
  def __init__(self,continuous=False):
    self._env = environment(continuous_control=continuous)
    self.n_step = 0
  def reset(self):
    self._env.reset_robot_position(random_=True)
    self._env.reset_target_position(random_=True)
    state = self._env.get_obs()
    return torch.tensor(state, dtype=torch.float32)

  def step(self, action):
    reward, done = self._env.step_(action.detach())
    state = self._env.get_obs()
    return torch.tensor(state, dtype=torch.float32), reward, done

  def render(self):
    return self._env.render()

  def terminate(self):
    self._env.terminate()
