from youBot_env import youBotEnv
from youBot_base import youBotBase
from youBot_arm import youBotArm
from pyrep.robots.arms.youBot import youBot
from pyrep.robots.mobiles.youbot import YouBot as youBot_base
import numpy as np
import torch
import random
from math import sqrt

class youBotAll(youBotArm, youBotBase):
    def __init__(self, scene_name,  obs_lowdim=True, rpa=6, reward_dense=True, boundary=1, demonstration_mode=False):
        super().__init__(scene_name, obs_lowdim=obs_lowdim, reward_dense=reward_dense, rpa=rpa, demonstration_mode=demonstration_mode, boundary=boundary)

        self.reward_dense = reward_dense
        self.action_space = 3 + 5
        self.action = [0 for _ in range(self.action_space)]

        self.arm.set_motor_locked_at_zero_velocity(1)

    def get_observation(self):
        if self.obs_lowdim:
            _, obsArm = youBotArm.get_observation(self)
            _, obsBase = youBotBase.get_observation(self)
            targ_vec = np.array(self.target_base.get_position()) - np.array(self.tip.get_position())
            return None, torch.tensor(np.concatenate((obsArm[:-11], obsBase[:-10], self.action, targ_vec),axis=0)).float()
        else:
            return env.render('arm'), torch.tensor(np.concatenate((obsArm[:16], obsBase[:6], self.action),axis=0)).float()

    def step(self,action):
        reward = 0
        self.action = action
        for _ in range(self.rpa):
            dem_action_base = youBotBase._set_actuation(self,action[:3])
            dem_action_arm = youBotArm._set_actuation(self,action[3:])

            if self.demonstration_mode:
                self.action = np.concatenate((dem_action_base, dem_action_arm), axis=0).tolist()

            self.prev_tip_pos = self.tip.get_position()

            pos_2d_prev = self.mobile_base.get_2d_pose()
            self.pr.step()
            pos_2d_next = self.mobile_base.get_2d_pose()

            for i in range(2):
                self.xy_vel[i] = ((pos_2d_next[i] - pos_2d_prev[i]) / 0.05)

            self.rot_vel[-1] = (pos_2d_prev[-1] - pos_2d_next[-1]) / 0.05

            reward_a, done = self._get_reward()
            reward += reward_a
            if reward_a == self.reward_termination:
                reward = self.reward_termination
                break
            elif done:
                break

        _, obs = self.get_observation()

        return obs, reward, done

    def reset(self):
        self.pr.set_configuration_tree(self.config_tree) # youBot model deteriorates over time so reset all dynamics at each new episode

        self._reset_target_position(random_=True)
        self._reset_base_position(random_=True)
        self._reset_arm(random_=False)
        self.pr.step()

        self.prev_tip_pos = np.array(self.tip.get_position())

        _, obs = self.get_observation()
        return obs

    def _get_reward(self):
        # Get the distance to target
        target_rel_pos = self.target_base.get_position(self.tip)
        dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2 + target_rel_pos[2]**2)

        # Distance to measure if robot out of bound
        pos_ref = self.mobile_base.get_2d_pose()
        dist_from_origin = sqrt(pos_ref[0]**2 + pos_ref[1]**2)

        if dist_ee_target < 0.05: #0.035
            reward = self.reward_termination
            self.done = True
        elif dist_from_origin > self.boundary: # Out of bound reward for navigation
            self.done = True
            reward = -3
        else:
            reward = -dist_ee_target/5 if self.reward_dense else -1

        return reward, self.done

    def _reset_target_position(self,random_=False, position=[0.6,0.6,0.3]):
        if random_:
            x_T,y_T, _ = self.rand_bound()
            z_T = random.uniform(0.2,0.4)
        else:
            x_T, y_T, z_T = position

        self.target_base.set_position([x_T,y_T,z_T])
        self.done = False

    def step_limit(self):
        return 350
