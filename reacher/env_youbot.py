from pyrep import PyRep
from math import pi, sqrt
import random
import numpy as np
from os.path import dirname, join, abspath
import torch

class environment(object):
    def __init__(self, obs_lowdim=True, rpa=4):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'youbot.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        self.target = self.pr.get_object('target')
        self.base_ref = self.pr.get_dummy('youBot_ref')
        self.youBot = self.pr.get_object('youBot')
        self.camera = self.pr.get_vision_sensor('Vision_sensor')

        self.wheel_joint_handle = []
        joint_name = ['rollingJoint_fl','rollingJoint_rl','rollingJoint_rr','rollingJoint_fr']
        for joint in joint_name:
            print(joint)
            self.wheel_joint_handle.append(self.pr.get_joint(joint))
            
        self.rpa = rpa
        self.done = False
        self.obs_lowdim = obs_lowdim

        ForwBackVel_range = [-240,240]
        LeftRightVel_range = [-240,240]
        RotVel_range = [-240,240]
        self.xy_vel_range = []
        self.xy_vel = [0,0]

        self.move_base() #Set velocity to 0

    def move_base(self,forwBackVel=0,leftRightVel=0,rotVel=0):
        self.wheel_joint_handle[0].set_joint_target_velocity(-forwBackVel-leftRightVel-rotVel)
        self.wheel_joint_handle[1].set_joint_target_velocity(-forwBackVel+leftRightVel-rotVel)
        self.wheel_joint_handle[2].set_joint_target_velocity(-forwBackVel-leftRightVel+rotVel)
        self.wheel_joint_handle[3].set_joint_target_velocity(-forwBackVel+leftRightVel+rotVel)
        for _ in range(self.rpa):
            pos_prev = self.base_ref.get_position()
            self.pr.step()
            pos_next = self.base_ref.get_position()

            for i in range(2):
                self.xy_vel[i] = ((pos_next[i] - pos_prev[i]) / 0.05)

    def render(self):
        img = self.camera.capture_rgb() # (dim,dim,3)
        return img*256

    def get_observation(self):
        if self.obs_lowdim:
            obs = []
            or_ = self.base_ref.get_orientation()
            pos = self.base_ref.get_position()

            targ_vec = np.array(self.target.get_position()) - np.array(pos)
            return np.concatenate(([or_[0]], pos[0:2], self.xy_vel, targ_vec[0:2]),axis=0)
        else:
            return None # This camera is mounted on the robot arm not the top view camera

    def step(self,action):
        self.move_base(forwBackVel=action[0],leftRightVel=action[1],rotVel=action[2])
        target_pos = self.target.get_position()
        youbot_pos = self.base_ref.get_position()
        dist_ee_target = sqrt((youbot_pos[0] - target_pos[0])**2 + \
        (youbot_pos[1] - target_pos[1])**2)

        if dist_ee_target < 0.3:
            reward = 1
            self.done = True
        else:
            reward = -dist_ee_target/10

        state = self.get_observation()
        return state, reward, self.done

    def reset(self):
        self.reset_target_position(random_=True)
        self.reset_robot_position(random_=True)
        self.move_base()
        state = self.get_observation()
        return state

    def reset_target_position(self,random_=False, position=[0,0]):
        if random_:
            x_T,y_T = self.rand_bound()
        else:
            x_T,y_T = position

        self.target.set_position([x_T,y_T,0.0275])

    def reset_robot_position(self,random_=False, position=[0.5,0.5], orientation=0):
        if random_:
            x_L, y_L = self.rand_bound() # Make sure target and robot are away from each other?
            orientation = random.random()*2*pi
        else:
            x_L, y_L = position

        self.youBot.set_position([x_L,y_L,0.095750])
        self.youBot.set_orientation([-90*pi/180 if orientation<0 else:90*pi/180,orientation,-90*pi/180 if orientation<0 else:90*pi/180])

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()

    def sample_action(self):
        return [(3 * random.random() - 1.5) for _ in range(self.action_space())]

    def rand_bound(self):
        xy_min = 0
        xy_max = 0.95
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
