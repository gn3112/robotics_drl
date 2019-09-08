from youBot_env import youBotEnv
from youBot_base import youBotBase
from youBot_arm import youBotArm
from pyrep.robots.arms.youBot import youBot
from pyrep.robots.mobiles.youbot import YouBot as youBot_base
import numpy as np
import torch
import random
from math import sqrt
from torchvision import transforms as T
from pyrep.objects.shape import Shape

def resize(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((128,128)),
                        T.ToTensor()])
    return resize(np.uint8(a*255))

def resize_d(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((128,128)),
                        T.Grayscale(num_output_channels=1),
                        T.ToTensor()])
    return resize(np.uint8(a*255))

class youBotAll(youBotArm, youBotBase):
    def __init__(self, scene_name,  obs_lowdim=True, rpa=6, reward_dense=True, boundary=1, demonstration_mode=False):
        super().__init__(scene_name, obs_lowdim=obs_lowdim, reward_dense=reward_dense, rpa=rpa, demonstration_mode=demonstration_mode, boundary=boundary)

        self.reward_dense = reward_dense
        self.obs_lowdim = obs_lowdim
        self.action_space = 3 + 5
        self.action = [0 for _ in range(self.action_space)]
        self.prev_action = [0 for _ in range(self.action_space)]
        self.steps_ep = 0
        self.frames = 1
        self.prev_obs = []
        self.arm.set_motor_locked_at_zero_velocity(1)
        self.table = Shape('table')
        self.training = True

    def get_observation(self):
        _, obsArm = youBotArm.get_observation(self)
        _, obsBase = youBotBase.get_observation(self)
        targ_vec = np.array(self.target_base.get_position()) - np.array(self.tip.get_position())
        targ_vec_base = np.array(self.target_base.get_position()[:2]) - np.array(self.mobile_base.get_2d_pose()[:2])
        if self.obs_lowdim:
            return None, torch.tensor(np.concatenate((obsArm[:-11], obsBase[:-10], self.action, targ_vec, targ_vec_base),axis=0)).float()
        else:
            new_obs = torch.cat((resize(self.render('arm')).view(-1,128,128),resize_d(self.camera_arm.capture_depth()).view(-1,128,128)), dim=0)
            if self.frames < 2:
                obs = new_obs
            else:
                if self.steps_ep == 0:
                    obs = new_obs.view(-1,128,128)
                    for _ in range(self.frames-1):
                        obs = torch.cat((obs,new_obs.view(-1,128,128)),dim=0)
                else:
                    obs = self.prev_obs
                    for i in range(self.frames-1):
                        obs[i*4:i*4+4,:,:] = obs[i*4+4:i*4+8,:,:]
                    obs[self.frames*4-4:self.frames*4,:,:] = new_obs

                    self.prev_obs = obs

            return obs.view(-1,128,128), torch.tensor(np.concatenate((obsArm, obsBase, self.action, targ_vec, targ_vec_base),axis=0)).float()

    def step(self,action):
        reward = 0
        self.action = action
        self.steps_ep += 1
        for _ in range(self.rpa):
            dem_action_base = youBotBase._set_actuation(self,action[:3])
            dem_action_arm = youBotArm._set_actuation(self,action[3:])

            if self.demonstration_mode:
                self.action = np.concatenate((dem_action_base, dem_action_arm), axis=0).tolist()

            pos_2d_prev = self.mobile_base.get_2d_pose()
            self.pr.step()
            pos_2d_next = self.mobile_base.get_2d_pose()

            self.prev_tip_pos = self.tip.get_position()

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

        self.prev_action = action
        img, obs = self.get_observation()
        if not self.obs_lowdim:
            obs = {'high': img, 'low': obs}
        return obs, reward, done

    def reset(self):
        self.steps_ep = 0

        # youBotArm._reset_target_position(self, random_=False, position=[-0.5,0,0.3])
        self._reset_target_position(random_=False)

        alpha = 1
        steps = 0
        while True:
            while alpha > 0.72:
                self.pr.set_configuration_tree(self.config_tree) # youBot model deteriorates over time so reset all dynamics at each new episode
                steps += 1
                self._reset_base_position(random_=True)

                camera_pos = np.array(self.camera_arm.get_position()[:2])
                mobile_pos = np.array(self.mobile_base.get_2d_pose()[:2])
                target_pos = np.array(self.target_base.get_position()[:2])

                vec1 = mobile_pos - camera_pos
                vec2 = target_pos - mobile_pos
                alpha = np.abs(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))
                if steps > 20:
                    break
            # self._reset_base_position(random_=False)
            self._reset_arm(random_=True)
            self.pr.step()

            mobile_orient = self.mobile_base.get_orientation()
            if mobile_orient[0] < 0.02 and mobile_orient[1] < 0.02:
                break

        self.prev_tip_pos = np.array(self.tip.get_position())

        img, obs = self.get_observation()
        if not self.obs_lowdim:
            obs = {'high': img, 'low': obs}
        return obs

    def eval():
        self.training = False

    def train():
        self.training = True

    def _get_reward(self):
        # Get the distance to target
        target_rel_pos = self.target_base.get_position(relative_to=self.tip)
        dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2 + target_rel_pos[2]**2)

        # Distance to measure if robot out of bound
        pos_ref = self.mobile_base.get_2d_pose()
        dist_from_origin = sqrt(pos_ref[0]**2 + pos_ref[1]**2)

        if dist_ee_target < 0.05: #0.035
            reward = self.reward_termination
            self.done = True
        #elif dist_from_origin > self.boundary: # Out of bound reward for navigation
        #    self.done = True
        #    reward = -3
        else:
            reward_act = (-np.sum(np.array(self.action) - np.array(self.prev_action))**2) / 20
            reward = -dist_ee_target/5 if self.reward_dense else 0
            reward += reward_act
        return reward, self.done

    def _reset_target_position(self,random_=False, position=[-1.15,0,0.325]):
        if random_:
            x_T,y_T, _ = self.rand_bound()
            #z_T = random.uniform(0.2,0.4)
            z_T = 0.325
        else:
            x_T, y_T, z_T = position

        #self.target_base.set_position([x_T,y_T,z_T])
        #self.table.set_position([x_T,y_T,0.15])
        self.done = False

    def step_limit(self):
        return 350
