from robot_env import robotEnv
from pyrep.robots.mobiles.robot import robot as robot_base
import numpy as np
import torch
from math import sqrt, pi, radians
import random

class robotBase(robotEnv):
    def __init__(self, scene_name, obs_lowdim=True, rpa=6, reward_dense=True, boundary=1, demonstration_mode=False):
        super().__init__(scene_name, reward_dense, boundary)
        # Base init and handles
        self.mobile_base = robot_base()
        self.target_base = self.mobile_base.target_base
        self.config_tree = self.mobile_base.get_configuration_tree()

        self.action_space = 3
        self.action = [0 for _ in range(self.action_space)]
        self.done = False

        self.obs_lowdim = obs_lowdim
        self.rpa = rpa
        self.demonstration_mode = demonstration_mode

        self.xy_vel = [0,0]
        self.rot_vel = [0]

        self.prev_error = np.array([0.,0.,0.])
        self.cumulative_error = np.array([0.,0.,0.])
        self.Kp = 1.
        self.Kd = 0
        self.Ki = 0

        # "Might need to remove this if consider a fixed camera pos/orient only"
        # if self.obs_lowdim == False and base == True:
        #     self.arm.set_joint_positions([self.arm_start_pos[0],self.arm_start_pos[1]
        #     ,self.arm_start_pos[2],0.35,self.arm_start_pos[4]])
        #     self.arm.set_motor_locked_at_zero_velocity(1)
        #     self.arm.set_joint_target_velocities([0,0,0,0,0])

    def get_observation(self):
        pos_2d = self.mobile_base.get_2d_pose()
        if self.obs_lowdim:
            targ_vec = np.array(self.target_base.get_position()[:2]) - np.array(self.mobile_base.get_2d_pose()[:2])
            return None, torch.tensor(np.concatenate((pos_2d, self.xy_vel, self.rot_vel, self.action, targ_vec),axis=0)).float() # removed prev actions
        else:
            return self.render('arm'), torch.tensor(np.concatenate((pos_2d, self.xy_vel, self.rot_vel),axis=0)).float()

    def step(self, action):
        reward = 0
        self.action = action
        for _ in range(self.rpa):
            self._set_actuation(action)

            pos_2d_prev = self.mobile_base.get_2d_pose()
            self.pr.step()
            pos_2d_next = self.mobile_base.get_2d_pose()

            for i in range(2):
                self.xy_vel[i] = ((pos_2d_next[i] - pos_2d_prev[i]) / 0.05)

            self.rot_vel[-1] = (pos_2d_next[-1] - pos_2d_prev[-1]) / 0.05

            reward_a, done = self._get_reward()
            reward += reward_a

            if reward_a == self.reward_termination:
                reward = self.reward_termination
                break
            elif done:
                break

        img, obs = self.get_observation()
        if not self.obs_lowdim:
            obs = {'high': img, 'low': obs}
        return obs, reward, done

    def reset(self):
        self.pr.set_configuration_tree(self.config_tree) # youBot model deteriorates over time so reset all dynamics at each new episode

        self._reset_target_position(random_=True)
        self._reset_base_position(random_=True)

        _, obs = self.get_observation()
        return obs

    def _reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T,y_T, _ = self.rand_bound()
        else:
            x_T,y_T = position

        self.target_base.set_position([x_T,y_T,0.3])
        self.done = False

    def _reset_base_position(self,random_=False, position=[0,0], orientation=radians(90)):
        if random_:
            target_pos = self.target_base.get_position()

            x_L, y_L, orientation = self.rand_bound()
            while sqrt((target_pos[0]-x_L)**2 + (target_pos[1]-y_L)**2) < 0.5:
                x_L, y_L, orientation = self.rand_bound()
        else:
            x_L, y_L = position

        for _ in range(4):
            self.mobile_base.set_base_angular_velocites([0,0,0])
            self.pr.step()

        self.mobile_base.set_2d_pose([x_L,y_L,orientation])
        self.pr.step()

    def _get_reward(self):
        target_rel_pos = self.target_base.get_position(self.mobile_base)
        dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2) - 0.3

        # Distance to measure if robot out of bound
        pos_ref = self.mobile_base.get_2d_pose()
        dist_from_origin = sqrt(pos_ref[0]**2 + pos_ref[1]**2)

        if dist_ee_target < 0.1: #0.035
            reward = self.reward_termination
            self.done = True
        elif dist_from_origin > self.boundary: # Out of bound reward for navigation
            self.done = True
            reward = -dist_ee_target/5 if self.reward_dense else -1
        else:
            reward = -dist_ee_target/5 if self.reward_dense else -1

        return reward, self.done

    def _set_actuation(self, action):
        scaled_action = [0,0,0]
        if not self.demonstration_mode:
            for i in range(2):
                scaled_action[i] = action[i]*2 #unnormalise by multiplying by 0.01 (max) for v=4rad/s
            scaled_action[2] = action[2]*2.5 #max v rota = 6 rad/s

            # e = np.array(scaled_action)
            # e_P = e
            # e_D = e - self.prev_error
            # e_I = self.cumulative_error + e
            # action[0] = self.Kp * e_P[0]  + self.Kd * e_D[0] + self.Ki * e_I[0]
            # action[1] = self.Kp * e_P[1]  + self.Kd * e_D[1] + self.Ki * e_I[1]
            # action[2] = self.Kp * e_P[2]  + self.Kd * e_D[2] + self.Ki * e_I[2]
            #
            # self.prev_error = e
            # self.cumulative_error = self.cumulative_error + e

            self.mobile_base.set_base_angular_velocites(scaled_action)
            return action
        else:
            for i in range(2):
                # scaled_action[i] = (action[i] * (0.05*0.1/2)) / 0.01
                scaled_action[i] = action[i] / 2
            scaled_action[2] = action[2] / 2.5
            self.action = scaled_action
            return scaled_action
