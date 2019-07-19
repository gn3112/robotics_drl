from pyrep import PyRep
from pyrep.robots.arms.youBot import youBot
from pyrep.robots.mobiles.youBot import youBot as youBot_base
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from math import pi, sqrt
import random
import numpy as np
from os.path import dirname, join, abspath
import torch
import time

class environment(object):
    def __init__(self, scene_name='youbot_navig.ttt', manipulator=False, base=True, obs_lowdim=True, rpa=6, reward_dense=True, boundary=1, demonstration_mode=False):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), scene_name)
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        if scene_name != 'youbot_navig.ttt':
            self.pr.import_model('youbot.ttm')

        # Arm init and handles
        self.arm = youBot()
        self.arm_start_pos = self.arm.get_joint_positions()
        self.tip = self.arm.get_tip()

        # Base init and handles
        self.mobile_base = youBot_base()
        self.target_base = self.mobile_base.target_base
        self.config_tree = self.mobile_base.get_configuration_tree()


        # Vision sensor handles
        self.camera_top = VisionSensor('Vision_sensor')
        self.camera_arm = VisionSensor('Vision_sensor0')

        # Environment parameters
        self.manipulator_active = manipulator
        self.base_active = base
        self.demonstration_mode = demonstration_mode
        self.rpa = rpa # Repeat action
        self.done = False
        self.obs_lowdim = obs_lowdim # Bool for high or low state space
        self.action = [0 for _ in range(self.action_space())]
        self.prev_tip_pos = np.array(self.tip.get_position())
        self.xy_vel = [0,0]
        self.rot_vel = [0]
        self.reward_dense = reward_dense
        self.reward_termination = 1 if self.reward_dense else 0
        self.boundary = boundary

        self.arm.set_motor_locked_at_zero_velocity(1)
        self.arm.set_joint_target_velocities([0,0,0,0,0])

        # Set correct orientation when camera is fixed on the arm
        "Might need to remove this if consider a fixed camera pos/orient only"
        if self.obs_lowdim == False and base == True:
            self.arm.set_joint_positions([self.arm_start_pos[0],self.arm_start_pos[1]
            ,self.arm_start_pos[2],0.35,self.arm_start_pos[4]])
            self.arm.set_motor_locked_at_zero_velocity(1)
            self.arm.set_joint_target_velocities([0,0,0,0,0])

    def render(self,view='top'):
        if view == 'top':
            img = self.camera_top.capture_rgb() # (dim,dim,3)
        elif view == 'arm':
            img = self.camera_arm.capture_rgb()
        return img*256

    def _get_observation_manipulator(self):
        if self.obs_lowdim:
            arm_joint_pos = self.arm.get_joint_positions()
            arm_joint_vel = self.arm.get_joint_velocities()
            tip_pos = self.tip.get_position()
            tip_or = self.tip.get_orientation()
            return np.concatenate((arm_joint_pos, arm_joint_vel, tip_pos, tip_or), axis=0)
        else:
            return np.concatenate((arm_joint_pos, arm_joint_vel), axis=0)  # Compute tip pos with fk?

    def _get_observation_base(self):
        if self.obs_lowdim:
            pos_2d = self.mobile_base.get_2d_pose()

            return np.concatenate((pos_2d, self.xy_vel, self.rot_vel),axis=0) # removed prev actions
        else:
            return np.concatenate((self.xy_vel, self.rot_vel),axis=0)

    def get_observation(self):
        if self.manipulator_active:
            targ_vec = np.array(self.target_base.get_position()) - np.array(self.tip.get_position())
        else:
            targ_vec = np.array(self.target_base.get_position()[:2]) - np.array(self.mobile_base.get_2d_pose()[:2])

        if self.base_active and self.manipulator_active:
            return env.render('arm') if not self.obs_lowdim else None, torch.tensor(np.concatenate((self._get_observation_manipulator(), self._get_observation_base(), targ_vec),axis=0))
        elif self.base_active:
            return env.render('arm') if not self.obs_lowdim else None, torch.tensor(np.concatenate((self._get_observation_base(),targ_vec),axis=0))
        elif self.manipulator_active:
            return env.render('arm') if not self.obs_lowdim else None, torch.tensor(np.concatenate((self._get_observation_manipulator(),targ_vec),axis=0))

    def set_actuaction(self,action):
        if not self.demonstration_mode:
            self.action = action

            # Rebound between -0.0157 and 0.0157 from -1 to 1 (tanh)
            # Actions are in the cartesian space

            if self.base_active and self.manipulator_active:
                for i in range(2):
                    action[i] = action[i]*0.01 #unnormalise by multiplying by 0.01 (max) for v=4rad/s
                action[2] = action[2]*6 #max v rota = 6 rad/s
                for i in range(3,6):
                    action[i] = action[i]*0.01 + self.prev_tip_pos[i-3]

                try:
                    joint_values_arm = self.arm.solve_ik(position=action[3:], euler=[0,0,1.57])
                    self.arm.set_joint_target_positions(joint_values_arm)
                except:
                    pass

                self.mobile_base.set_base_angular_velocites(action[:3])

            elif self.base_active:
                for i in range(2):
                    action[i] = action[i]*0.01 #unnormalise by multiplying by 0.01 (max) for v=4rad/s
                action[2] = action[2]*6 #max v rota = 6 rad/s
                self.mobile_base.set_cartesian_position(action)

            elif self.manipulator_active:
                for i in range(3):
                    action[i] = action[i]*0.01 + self.prev_tip_pos
                joint_values_arm = self.arm.solve_ik(position=action, euler=[0,0,0])
                self.arm.set_joint_target_positions(joint_values_arm)
        else:
            if self.base_active and self.manipulator_active:
                for i in range(2):
                    self.action[i] = (action[i] * (0.05*0.1/2)) / 0.01

                self.action[2] = action[2]/6

                tip_pos = np.array(self.tip.get_position())
                self.action[3:] = (tip_pos - self.prev_tip_pos)/0.01
            elif self.base_active:
                for i in range(2):
                    self.action[i] = (action[i] * (0.05*0.1/2)) / 0.01

                self.action[2] = action[2]/6
            elif self.manipulator_active:
                tip_pos = np.array(self.tip.get_position())
                self.action = (tip_pos - self.prev_tip_pos)/0.01

    def step(self,action):
        # Assert size action

        self.prev_tip_pos = self.tip.get_position()
        reward = 0
        for _ in range(self.rpa):
            self.set_actuaction(action)
            pos_2d_prev = self.mobile_base.get_2d_pose()
            self.pr.step()
            pos_2d_next = self.mobile_base.get_2d_pose()

            for i in range(2):
                self.xy_vel[i] = ((pos_2d_next[i] - pos_2d_prev[i]) / 0.05)

            self.rot_vel[-1] = (pos_2d_next[-1] - pos_2d_prev[-1]) / 0.05

            reward_a, done = self.get_reward()
            reward += reward_a
            if reward_a == self.reward_termination:
                reward = self.reward_termination
                break
            elif done:
                break

        _, obs = self.get_observation()

        return obs, reward, done

    def get_reward(self):

        # Get the distance to target
        if self.manipulator_active:
            target_rel_pos = self.target_base.get_position(self.tip)
            dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2 + target_rel_pos[2]**2)
        else:
            target_rel_pos = self.target_base.get_position(self.mobile_base)
            dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2)


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

    def reset(self):
        self.pr.set_configuration_tree(self.config_tree) # youBot model deteriorates over time so reset all dynamics at each new episode

        if self.base_active and self.manipulator_active:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)
            self.reset_arm()
        elif self.base_active:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)

        elif self.manipulator_active:
            self.reset_arm()

        _, obs = self.get_observation()
        return obs

    def reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T,y_T = self.rand_bound()
        else:
            x_T,y_T = position

        self.target_base.set_position([x_T,y_T,0.3])
        self.done = False

    def reset_base_position(self,random_=False, position=[0.5,0.5], orientation=0):
        if random_:
            target_pos = self.target_base.get_position()

            x_L, y_L = self.rand_bound()
            while sqrt((target_pos[0]-x_L)**2 + (target_pos[1]-y_L)**2) < 0.5:
                x_L, y_L = self.rand_bound()

            # col = self.mobile_base.assess_collision()
            #
            # if col:
            #     print("Collision detected")

            orientation = random.random()*2*pi

        else:
            x_L, y_L = position

        # Reset velocity to 0
        for _ in range(4):
            self.mobile_base.set_base_angular_velocites([0,0,0])
            self.pr.step()

        self.mobile_base.set_2d_pose([x_L,y_L,orientation])
        self.pr.step()

    def reset_arm(self):
        self.arm.set_joint_positions(self.arm_start_pos)

        for _ in range(4):
            self.arm.set_joint_target_velocities([0,0,0,0,0])
            self.pr.step()

        # Need joint intervals
        # Need a collision check

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()

    def sample_action(self):
        return [(2 * random.random() - 1) for _ in range(self.action_space())]

    def rand_bound(self):
        xy_min = 0
        xy_max = self.boundary - 0.2
        x = random.random()*(xy_max-xy_min) + xy_min
        y_max = sqrt(xy_max**2-x**2)
        y_min = 0
        y = random.random()*(y_max-y_min) + y_min

        quadrant = random.randint(1,4)
        if quadrant == 1:
            x = -x
            y = -y
        elif quadrant == 2:
            x = -x
            y = y
        elif quadrant == 3:
            x = x
            y = -y
        elif quadrant == 4:
            x = x
            y = y
        return x,y

    def action_space(self):
        if self.base_active and self.manipulator_active:
            return 6
        elif self.manipulator_active:
            return 3
        elif self.base_active:
            return 3

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        _, obs = self.get_observation()
        return obs.shape

    def action_boundaries(self):
        return [[-240,240],[-240,240],[-240,240]]

    def step_limit(self):
        return 200
