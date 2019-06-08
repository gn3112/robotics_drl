" Define an environment and build utilities to get state, reward, action..."
from pyrep import PyRep
from math import sqrt, pi, exp, cos, sin
from matplotlib import pyplot as plt
import random
import torch
from os.path import dirname, join, abspath
import numpy as np

class environment(object):
    def __init__(self,continuous_control=False):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'reacher_v2.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        self.reached = 0
        self.done = False
        self.continuous_control = continuous_control
        self.target = self.pr.get_object('target')
        self.end_effector = self.pr.get_dummy('end_effector')
        self.joint1 = self.pr.get_joint('joint_1')
        self.joint2 = self.pr.get_joint('joint_2')
        self.reacher = self.pr.get_object('reacher')
        self.camera = self.pr.get_vision_sensor('Vision_sensor')
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
                ee_pos = self.end_effector_pos()
                dist_ee_target = sqrt((ee_pos[0] - self.target_pos[0])**2 + \
                (ee_pos[1] - self.target_pos[1])**2)
                if dist_ee_target < 0.054:
                    reward = 1
                    print('TARGET REACHED')
                    self.done = True
                    break
                else:
                    reward = -dist_ee_target/10
                print('Reward:%s'%reward)

    def render(self):
        img = self.camera.capture_rgb()
        return img*256

    def get_obs(self):
        joints_pos = self.get_joints_pos()
        cos_joints = []
        sin_joints = []
        for _, theta in enumerate(joints_pos):
            cos_joints.append(cos(theta))
            sin_joints.append(sin(theta))
        joints_vel = self.get_joints_vel()
        target_pos = self.target_position()
        ee_pos = self.end_effector_pos()
        targ_vec = np.array(ee_pos) - np.array(target_pos)
        obs = np.concatenate((cos_joints,sin_joints,joints_pos,joints_vel,targ_vec[0:2]),axis=0)
        return obs

    def step(self,action):
        #TODO: change directly from pos to vel without changing scene
        for action_rep in range(1):
            if self.continuous_control == True:
                self.joint1.set_joint_target_velocity(action[0]) # radians/s
                self.joint2.set_joint_target_velocity(action[1])
            
            else:
                position_all = self.action_all[action]
                joints_pos = self.get_joints_pos()
                joint1_pos = joints_pos[0]
                joint2_pos = joints_pos[1]
                self.joint1.set_joint_target_position(joint1_pos + position_all[0]) # radians
                self.joint2.set_joint_target_position(joint2_pos + position_all[1])

            self.pr.step()
        
        ee_pos = self.end_effector_pos()
        dist_ee_target = sqrt((ee_pos[0] - self.target_pos[0])**2 + \
        (ee_pos[1] - self.target_pos[1])**2)

        if dist_ee_target < 0.054:
            reward = 1
            self.done = True
        else:
            reward = -dist_ee_target/10

        state = self.get_obs()
        return torch.tensor(state, dtype=torch.float32), reward, self.done


    def end_effector_pos(self):
        return self.end_effector.get_position()

    def target_position(self):
        return self.target.get_position()

    def get_joints_pos(self):
        self.joint1_pos = self.joint1.get_joint_position()
        self.joint2_pos = self.joint2.get_joint_position()
        return [self.joint1_pos*(180/3.14),self.joint2_pos*(180/3.14)]

    def get_joints_vel(self):
        self.joint1_vel = self.joint1.get_joint_velocity()
        self.joint2_vel = self.joint2.get_joint_velocity()
        return [self.joint1_vel,self.joint2_vel]

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
        # self.pr.step()
        self.target_pos = self.target_position()
        self.done = False

    def reset_robot_position(self,random_=False, joint1_pos=0, joint2_pos=0):
        if random_ == True:
            joint1_pos = random.random()*2*pi
            joint2_pos = random.random()*2*pi   
        
        if self.continuous_control:
            [self.step([0,0]) for _ in range(5)]
        
        self.joint1._set_joint_position_torque_force_mode(joint1_pos) # radians
        self.joint2._set_joint_position_torque_force_mode(joint2_pos)
        self.pr.step()
    
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
                reward = self.step_(action)

                steps += 1
                if steps == 40:
                    break

                if reward == 1:
                    steps_all.append(steps)
                    break

        return sum(steps_all)/episodes

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()

    def reset(self):
        self.reset_target_position(random_=True)
        self.reset_robot_position(random_=False)
        state = self.get_obs()
        return torch.tensor(state, dtype=torch.float32)
