from pyrep import PyRep
from pyrep.robots.arms import youBot
from math import pi, sqrt
import random
import numpy as np
from os.path import dirname, join, abspath
import torch
import time

class environment(object):
    def __init__(self, manipulator=False, base=True, obs_lowdim=True, rpa=6):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'youbot.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()
        time.sleep(0.1)

        self.arm = self.pr.get_arm(youBot)
        self.arm_start_pos = self.arm.get_joint_positions()

        self.wheel = []
        self.slipping = []
        for lr in ['fl','rl','rr','fr']:
            self.wheel.append(self.pr.get_object('wheel_respondable_%s'%lr))
            self.slipping.append(self.pr.get_object('slippingJoint_%s'%lr))

        self.target = self.pr.get_object('target')
        self.base_ref = self.pr.get_dummy('youBot_ref')
        self.tip = self.pr.get_dummy('youBot_tip')
        self.youBot = self.pr.get_object('youBot')
        self.camera = self.pr.get_vision_sensor('Vision_sensor')
        self.collection_arm = self.pr.get_object('youBot_arm')

        self.wheel_joint_handle = []
        joint_name = ['rollingJoint_fl','rollingJoint_rl','rollingJoint_rr','rollingJoint_fr']
        for joint in joint_name:
            self.wheel_joint_handle.append(self.pr.get_joint(joint))


        self.manipulator = manipulator
        self.base = base
        self.rpa = rpa
        self.done = False
        self.obs_lowdim = obs_lowdim
        self.action = [0,0,0]

    def reset_wheel(self):
        p = [[-pi/4,0,0],[pi/4,0,pi/4]]
        for i in range(4):
            self.wheel[i].reset_dynamic_object()
            self.slipping[i].set_position([0,0,0],relative_to=self.wheel_joint_handle[i],reset_dynamics=True)
            self.slipping[i].set_orientation(p[1 if i > 1 else 0],relative_to=self.wheel_joint_handle[i],reset_dynamics=True)
            self.wheel[i].set_position([0,0,0],relative_to=self.wheel_joint_handle[i],reset_dynamics=True)
            self.wheel[i].set_orientation([0,0,0],relative_to=self.wheel_joint_handle[i],reset_dynamics=True)

    def move_base(self,forwBackVel=0,leftRightVel=0,rotVel=0):
        self.reset_wheel()
        self.wheel_joint_handle[0].set_joint_target_velocity(-forwBackVel-leftRightVel-rotVel)
        self.wheel_joint_handle[1].set_joint_target_velocity(-forwBackVel+leftRightVel-rotVel)
        self.wheel_joint_handle[2].set_joint_target_velocity(-forwBackVel-leftRightVel+rotVel)
        self.wheel_joint_handle[3].set_joint_target_velocity(-forwBackVel+leftRightVel+rotVel)

    def move_manipulator(self,joints_vel=[0,0,0,0,0]):
        self.arm.set_joint_target_velocities(joint_vel)

    def render(self):
        img = self.camera.capture_rgb() # (dim,dim,3)
        return img*256

    def get_observation_manipulator(self):
        arm_joint_pos = self.arm.get_joint_positions()
        arm_joint_vel = self.arm.get_joint_velocities()
        tip_pos = self.tip.get_position()
        tip_or = self.tip.get_orientation()
        return np.concatenate((arm_joint_pos, arm_joint_vel, tip_pos, tip_or), axis=0)

    def get_observation_base(self):
        if self.obs_lowdim:
            obs = []
            or_ = self.base_ref.get_orientation()
            pos = self.base_ref.get_position()

            targ_vec = np.array(self.target.get_position()) - np.array(pos)
            return np.concatenate(([or_[0]], pos[0:2], self.xy_vel, self.action, targ_vec[0:2]),axis=0)
        else:
            return None # This camera is mounted on the robot arm not the top view camera

    def get_observation(self):
        if self.base and self.manipulator:
            return np.concatenate(self.get_observation_manipulator(), self.get_observation_base())
        elif self.base:
            return self.get_observation_base()
        elif self.manipulator:
            return self.get_observation_manipulator()

    def step(self,action):
        self.action = action

        if self.base and self.manipulator:
            self.move_base(forwBackVel=action[0],leftRightVel=action[1],rotVel=action[2])
            self.move_manipulator(joints_vel=action[3])
        elif self.base:
            self.move_base(forwBackVel=action[0],leftRightVel=action[1],rotVel=action[2])
        elif self.manipulator:
            self.move_manipulator(joints_vel=action[0])

        for _ in range(self.rpa):
            pos_prev = self.base_ref.get_position()
            self.pr.step()
            pos_next = self.base_ref.get_position()

            for i in range(2):
                self.xy_vel[i] = ((pos_next[i] - pos_prev[i]) / 0.05)

        target_pos = self.target.get_position()
        youbot_pos = self.base_ref.get_position()
        dist_ee_target = sqrt((youbot_pos[0] - target_pos[0])**2 + \
        (youbot_pos[1] - target_pos[1])**2)

        pos_ref = self.base_ref.get_position()
        dist_from_origin = sqrt(pos_ref[0]**2 + pos_ref[1]**2)

        if dist_ee_target < 0.35:
            reward = 1
            self.done = True
        elif dist_from_origin > 2.4:
            self.done = True
            reward = -dist_ee_target/3
        else:
            reward = -dist_ee_target/3

        state = self.get_observation()
            return state, reward, self.done

    def reset(self):
        if self.base and self.manipulator:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)
            self.reset_arm()
        elif self.base:
            self.reset_target_position(random_=True)
            self.reset_base_position(random_=True)

        elif self.manipulator:
            self.reset_arm()

        state = self.get_observation()
        return state

    def reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T,y_T = self.rand_bound()
        else:
            x_T,y_T = position

        self.target.set_position([x_T,y_T,0.0275])
        self.done = False

    def reset_base_position(self,random_=False, position=[0.5,0.5], orientation=0):
        if random_:
            target_pos = self.target.get_position()
            x_L, y_L = self.rand_bound()
            while abs(sqrt(x_L**2 + y_L**2) - sqrt(target_pos[0]**2 + target_pos[1]**2)) < 0.5:
                x_L, y_L = self.rand_bound()
            orientation = random.random()*2*pi
        else:
            x_L, y_L = position

        self.youBot.set_position([x_L,y_L,0.095750])
        self.youBot.set_orientation([-90*pi/180 if orientation<0 else 90*pi/180,orientation,-90*pi/180 if orientation<0 else 90*pi/180])
        for _ in range(4):
            self.move_base()
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
        return [(2*240*pi/180 * random.random() - 240*pi/180) for _ in range(self.action_space())]

    def rand_bound(self):
        xy_min = 0
        xy_max = 1.4
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
        return 3

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        return self.get_observation().shape

    def action_boundaries(self):
        return [[-240,240],[-240,240],[-240,240]]

    def step_limit(self):
        return 160
