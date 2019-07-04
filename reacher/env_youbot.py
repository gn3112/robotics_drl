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
    def __init__(self, manipulator=False, base=True, obs_lowdim=True, rpa=1, demonstration_mode=False):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'youbot.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        # Arm init and handles
        self.arm = youBot()
        self.arm_start_pos = self.arm.get_joint_positions()
        self.tip = self.arm.get_tip()

        # Base init and handles
        self.mobile_base = youBot_base()
        self.base_ref = self.mobile_base.base_ref
        self.target_base = self.mobile_base.target_base

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
        self.action = [0,0,0]
        self.xy_vel = [0,0]
        self.rot_vel = [0]

        # Set correct orientation when camera is fixed on the arm
        if self.obs_lowdim == False and base == True:
            self.arm.set_joint_positions([0,0,0,0.35,0])

    def render(self,view='top'):
        if view == 'top':
            img = self.camera_top.capture_rgb() # (dim,dim,3)
        elif view == 'arm':
            img = self.camera_arm.capture_rgb()
        return img*256

    def _get_observation_manipulator(self):
        arm_joint_pos = self.arm.get_joint_positions()
        arm_joint_vel = self.arm.get_joint_velocities()
        tip_pos = self.tip.get_position()
        tip_or = self.tip.get_orientation()
        return np.concatenate((arm_joint_pos, arm_joint_vel, tip_pos, tip_or), axis=0)

    def _get_observation_base(self):
        if self.obs_lowdim:
            or_ = self.mobile_base.get_base_orientation()
            pos = self.mobile_base.get_base_position()

            targ_vec = np.array(self.target_base.get_position()) - np.array(pos)
            return np.concatenate(([or_], pos[0:2], self.xy_vel, self.rot_vel, self.action, targ_vec[0:2]),axis=0)
        else:
            return self.render(view='arm') # This camera is mounted on the robot arm not the top view camera

    def get_observation(self):
        if self.base_active and self.manipulator_active:
            return torch.tensor(np.concatenate((self._get_observation_manipulator(), self._get_observation_base()),axis=0))
        elif self.base_active:
            return torch.tensor(self._get_observation_base())
        elif self.manipulator_active:
            return torch.tensor(self._get_observation_manipulator())

    def move_manipulator(self,joints_vel=[0,0,0,0,0]):
        self.arm.set_joint_target_velocities(joints_vel)

    def step(self,action):

        if not self.demonstration_mode:
            self.action = action.cpu() # Change here
            # Rebound between -0.0157 and 0.0157 from -1 to 1 (tanh)
            for i in range(2): #TODO: Add option for action type
                action[i] = action[i]*0.01# unnormalise by multiplying by 0.01 (max) for v=4rad/s
            action[-1] = action[-1]*6 # max v rota = 6 rad/s

            if self.base_active and self.manipulator_active:
                self.mobile_base.set_base_angular_velocites(action[:3])
                self.move_manipulator(joints_vel=action[3:])

            elif self.base_active:
                self.mobile_base.set_cartesian_position(action)

            elif self.manipulator_active:
                self.move_manipulator(joints_vel=action)
        else:
            for i in range(2): #TODO: Add option for action type
                self.action[i] = (action[i] * (0.05*0.1/2)) / 0.01

            self.action[-1] = action[-1]/6

        for _ in range(self.rpa):
            pos_prev = self.mobile_base.get_base_position()
            rot_prev = self.mobile_base.get_base_orientation()
            self.pr.step()
            pos_next = self.mobile_base.get_base_position()
            rot_next = self.mobile_base.get_base_orientation()

            for i in range(2):
                self.xy_vel[i] = ((pos_next[i] - pos_prev[i]) / 0.05)

            self.rot_vel[-1] = (rot_prev - rot_next) / 0.05

        reward, done = self.get_reward()

        return self.get_observation(), reward, done

    def get_reward(self):
        # Get the distance to target
        target_pos = self.target_base.get_position()
        youbot_pos = self.mobile_base.get_base_position()
        dist_ee_target = sqrt((youbot_pos[0] - target_pos[0])**2 + \
        (youbot_pos[1] - target_pos[1])**2)

        pos_ref = self.base_ref.get_position()
        dist_from_origin = sqrt(pos_ref[0]**2 + pos_ref[1]**2)

        #TODO: Add reward function for the arm and generic one for both arm and base
        if dist_ee_target < 0.35:
            reward = 1
            self.done = True
        elif dist_from_origin > 2.4: # Out of bound reward for navigation
            self.done = True
            reward = -dist_ee_target/3
        else:
            reward = -dist_ee_target/3

        return reward, self.done

    def reset(self):
        if self.base_active and self.manipulator_active:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)
            self.reset_arm()
        elif self.base_active:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)

        elif self.manipulator_active:
            self.reset_arm()

        return self.get_observation()

    def reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T,y_T = self.rand_bound()
        else:
            x_T,y_T = position

        self.target_base.set_position([x_T,y_T,0.15])
        self.done = False

    def reset_base_position(self,random_=False, position=[0.5,0.5], orientation=0):
        if random_:
            target_pos = self.target_base.get_position()
            x_L, y_L = self.rand_bound()
            while abs(sqrt(x_L**2 + y_L**2) - sqrt(target_pos[0]**2 + target_pos[1]**2)) < 0.35:
                x_L, y_L = self.rand_bound()
            orientation = random.random()*2*pi
        else:
            x_L, y_L = position

        self.mobile_base.set_base_position([x_L,y_L])
        self.mobile_base.set_base_orientation(angle=orientation)

        # Reset velocity to 0
        for _ in range(4):
            self.mobile_base.set_base_angular_velocites([0,0,0])
            self.pr.step()

    def reset_arm(self):
        self.arm.set_joint_positions(self.arm_start_pos)

        for _ in range(4):
            self.move_manipulator()
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
        xy_max = 1.2
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
        if self.base_active:
            return 3
        elif self.manipulator_active:
            return 5
        else:
            return 8

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        return self.get_observation().shape

    def action_boundaries(self):
        return [[-240,240],[-240,240],[-240,240]]

    def step_limit(self):
        return 200
