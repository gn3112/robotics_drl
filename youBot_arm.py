from youBot_env import youBotEnv
from pyrep.robots.arms.youBot import youBot
from pyrep.robots.mobiles.youbot import YouBot as youBot_base
from pyrep.objects.dummy import Dummy
import torch
import numpy as np
import random
from math import sqrt

class youBotArm(youBotEnv):
    def __init__(self, scene_name, obs_lowdim=True, rpa=6, reward_dense=True, demonstration_mode=False, boundary=0):
        super().__init__(scene_name, reward_dense, boundary)

        # Arm init and handles
        self.arm = youBot()
        self.mobile_base = youBot_base()
        self.arm_start_pos = self.arm.get_joint_positions()
        self.tip = self.arm.get_tip()
        self.config_tree = self.arm.get_configuration_tree()

        self.action_space = 5
        self.prev_tip_pos = np.array(self.tip.get_position())
        self.action = [0 for _ in range(self.action_space)]
        self.done = False

        self.obs_lowdim = obs_lowdim
        self.rpa = rpa
        self.demonstration_mode = demonstration_mode

        self.target_base = Dummy('youBot_target_base')

        self.arm.set_motor_locked_at_zero_velocity(1)
        self.arm.set_control_loop_enabled(0)
        self.arm.set_joint_target_velocities([0,0,0,0,0])
        self.mobile_base.set_motor_locked_at_zero_velocity(1)
        self.mobile_base.set_joint_target_velocities([0,0,0,0])

    def get_observation(self):
        arm_joint_pos = self.arm.get_joint_positions()
        arm_joint_vel = self.arm.get_joint_velocities()
        if self.obs_lowdim:
            targ_vec = np.array(self.target_base.get_position()) - np.array(self.tip.get_position())
            tip_pos = self.tip.get_position()
            tip_or = self.tip.get_orientation()
            return None, torch.tensor(np.concatenate((arm_joint_pos, arm_joint_vel, tip_pos, self.action, targ_vec), axis=0)).float() # ADD tip_or when augmenting action space
        else:
            return env.render('arm'), torch.tensor(np.concatenate((arm_joint_pos, arm_joint_vel),axis=0)).float()

    def step(self, action):
        reward = 0
        self.action = action
        for _ in range(self.rpa):
            self._set_actuation(action)
            self.prev_tip_pos = self.tip.get_position()
            self.pr.step()

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
        self._reset_arm(random_=True)

        _, obs = self.get_observation()
        return obs

    def _reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T = random.uniform(0.1,0.23)
            y_T = random.uniform(-0.1, 0.1)
            z_T = random.uniform(0.2, 0.3)
        else:
            x_T,y_T = position
            z_T = 0.3

        self.target_base.set_position([x_T,y_T,z_T])
        self.done = False

    def _reset_arm(self, random_=False):
        if random_:
            while True:
                x = random.uniform(-0.4,0.08)
                y = random.uniform(-0.2,0.2)
                z = random.uniform(0.2,0.45)
                try:
                    joints_pos = self.arm.solve_ik(position=[x,y,z], euler=[0,0,1.57])
                except:
                    continue

                break

            self.arm.set_joint_positions(joints_pos)
        else:
            self.arm.set_joint_positions(self.arm_start_pos)

        for _ in range(4):
            self.arm.set_joint_target_velocities([0,0,0,0,0])
            self.pr.step()

    def _get_reward(self):
        # Get the distance to target
        target_rel_pos = self.target_base.get_position(self.tip)
        dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2 + target_rel_pos[2]**2)

        if dist_ee_target < 0.02:
            reward = self.reward_termination
            self.done = True
        else:
            reward = -dist_ee_target/5 if self.reward_dense else -1

        return reward, self.done

    def _set_actuation(self, action):
        if not self.demonstration_mode:
            # scaled_action = np.array(action)*0.01 + np.array(self.prev_tip_pos)
            # try:
            #     joint_values_arm = self.arm.solve_ik(position=scaled_action.tolist(), euler=[0,0,1.57])
            #     self.arm.set_joint_target_positions(joint_values_arm)
            # except:
            #     pass
            scaled_action = np.array(action) * 1.57
            self.arm.set_joint_target_velocities(scaled_action)
            return action
        else:
            # tip_pos = np.array(self.tip.get_position())
            # scaled_action = ((tip_pos - self.prev_tip_pos)/0.01).tolist()
            # self.action = scaled_action
            scaled_action = np.array(self.arm.get_joint_velocities())/1.57
            return scaled_action

    def step_limit(self):
        return 80
