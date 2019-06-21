" Define an environment and build utilities to get state, reward, action..."
from pyrep import PyRep
from pyrep.robots.arms import Reacher
from math import sqrt, pi, exp, cos, sin
from matplotlib import pyplot as plt
import random
import torch
from os.path import dirname, join, abspath
import numpy as np
from torchvision import transforms as T
from PIL import Image
from invert import Invert

def resize(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        Invert(),
                        T.ToTensor(),
                        T.Normalize((0.5,), (0.5,))])
    return resize(np.uint8(a))

class environment(object):
    def __init__(self,continuous_control=True, obs_lowdim=True, rpa=3, frames=4):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'reacher_v2.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        self.continuous_control = continuous_control
        self.target = self.pr.get_object('target')
        self.tip = self.pr.get_dummy('Reacher_tip')
        self.camera = self.pr.get_vision_sensor('Vision_sensor')
        self.agent = self.pr.get_arm(Reacher)

        self.done = False
        self.rpa = rpa
        self.obs_lowdim = obs_lowdim
        self.frames = frames
        self.prev_obs = []
        self.steps_ep = 0
        self.increment = 4*pi/180 # to radians
        self.action_all = [[self.increment,self.increment],
                      [-self.increment,-self.increment],
                      [0,self.increment],
                      [0,-self.increment],
                      [self.increment,0],
                      [-self.increment,0],
                      [-self.increment,self.increment],
                      [self.increment,-self.increment]]

    def threshold_check(self):
        for _ in range(5):
            self.reset_target_position(random_=True)
            while True:
                self.pr.step()
                tip_pos = self.tip_position()
                dist_tip_target = sqrt((tip_pos[0] - self.target_pos[0])**2 + \
                (tip_pos[1] - self.target_pos[1])**2)
                if dist_tip_target < 0.07:
                    reward = 1
                    self.done = True
                    break
                else:
                    reward = -dist_tip_target/5
                print('Reward:%s'%reward)

    def render(self):
        return self.camera.capture_rgb()*256 # (dim,dim,3)

    def get_observation(self):
        if self.obs_lowdim:
            joints_pos = self.agent.get_joint_positions()

            cos_joints = []
            sin_joints = []
            for theta in joints_pos:
                cos_joints.append(cos(theta))
                sin_joints.append(sin(theta))

            joints_vel = self.agent.get_joint_velocities()

            target_pos = self.target_position()
            tip_pos = self.tip_position()
            tip_target_vec = np.array(tip_pos) - np.array(target_pos)

            return np.concatenate((cos_joints, sin_joints, joints_pos, joints_vel, tip_target_vec[0:2]),axis=0)
       
        else:
            new_obs = resize(self.render()).view(-1,64,64)

            if self.steps_ep == 0:
                obs = new_obs.expand(self.frames,64,64)
            else:
                obs = self.prev_obs
                for i in range(self.frames-1):
                    obs[i,:,:] = obs[i+1,:,:]
                
                obs[self.frames-1,:,:] = new_obs
            
            self.prev_obs = obs
            return obs.view(-1,64,64)

    def step(self,action):
        self.steps_ep += 1
        #TODO: change directly from pos to vel without changing in scene
        for action_rep in range(self.rpa):
            if self.continuous_control == True:
                self.agent.set_joint_target_velocities(action) # radians/s

            else:
                position_all = self.action_all[action]
                joints_pos = self.agent.get_joint_positions()
                self.agent.set_joint_target_positions([joints_pos[0] + position_all[0], joints_pos[1] + position_all[1]]) # radians

            self.pr.step()

        tip_pos = self.tip_position()
        tip_target_dist = sqrt((tip_pos[0] - self.target_pos[0])**2 + \
        (tip_pos[1] - self.target_pos[1])**2)

        if tip_target_dist < 0.07:
            reward = 1
            self.done = True
            print('TARGET REACHED')
        else:
            reward = -tip_target_dist/3

        state = self.get_observation()

        return state, reward, self.done


    def tip_position(self):
        return self.tip.get_position()

    def target_position(self):
        return self.target.get_position()

    def reset_target_position(self,random_=False,x=0.4,y=0.4):
        if random_ == True:
            xy_min = 0.12
            xy_max = 0.5325
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

        self.target.set_position([x,y,0.0275])
        self.target_pos = self.target_position()
        self.done = False

    def reset_robot_position(self,random_=False, joints_pos=[0,0]):
        if random_ == True:
            joints_pos = [random.random()*2*pi for _ in range(2)]

        self.agent.set_joint_positions(joints_pos) # radians

        if self.continuous_control:
            for _ in range(2):
                self.agent.set_joint_target_velocities([0,0])
                self.pr.step()

    def sample_action(self):
        return [1.3*(3 * random.random() - 1.5),1.3*(3 * random.random() - 1.5)]


    def display(self):
        img = self.camera.capture_rgb()
        plt.imshow(img,interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.pause(0.01)

    def random_agent(self,episodes=10):
        steps_all = []
        for _ in range(episodes):
            steps = 0
            while True:
                action = random.randrange(len(self.action_all))
                reward = self.step(action)

                steps += 1
                if steps == 40:
                    break

                if reward == 1:
                    steps_all.append(steps)
                    break

        return sum(steps_all)/episodes

    def terminate(self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()

    def reset(self):
        self.steps_ep = 0
        self.reset_target_position(random_=True)
        self.reset_robot_position(random_=True)
        state = self.get_observation()
        return state

    def action_space(self):
        return 2

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        return self.get_observation().shape

    def step_limit(self):
        return 60
