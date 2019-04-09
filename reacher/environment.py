" Define an environment and build utilities to get state, reward, action..."
from pyrep import PyRep
from math import sqrt, pi, exp
from matplotlib import pyplot as plt
import random
from os.path import dirname, join, abspath

class environment(object):
    def __init__(self,position_control=True):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'reacher.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        self.reached = 0
        self.position_control = position_control
        self.target = self.pr.get_object('target')
        self.end_effector = self.pr.get_dummy('end_effector')
        self.joint1 = self.pr.get_joint('link_1')
        self.joint2 = self.pr.get_joint('link_2')
        self.reacher = self.pr.get_object('reacher')
        self.camera = self.pr.get_vision_sensor('Vision_sensor')

        self.increment = 2*pi/180 # to radians
        self.action_all = [[self.increment,self.increment],
                      [-self.increment,-self.increment],
                      [0,self.increment],
                      [0,-self.increment],
                      [self.increment,0],
                      [-self.increment,0],
                      [-self.increment,self.increment],
                      [self.increment,-self.increment],
                      [0,0]]

    def render(self):
        img = self.camera.capture_rgb()
        return img

    def step_(self,action):
        if self.position_control != True:
            velocity_all = self.action_all[action]
            #TO DO
            self.joint1.set_joint_target_velocity(velocity_all[0]) # radians/s
            self.joint2.set_joint_target_velocity(velocity_all[1])
        else:
            position_all = self.action_all[action]
            joints_pos = self.get_joints_pos()
            joint1_pos = joints_pos[0]
            joint2_pos = joints_pos[1]
            self.joint1.set_joint_target_position(joint1_pos + position_all[0]) # radians
            self.joint2.set_joint_target_position(joint2_pos + position_all[1])

        self.pr.step()
        ee_pos = self.end_effector_pos()
        # target_pos = self.target_position()
        dist_ee_target = sqrt((ee_pos[0] - self.target_pos[0])**2 + \
        (ee_pos[1] - self.target_pos[1])**2)
        # self.pos_target = self.target.get_position()
        # self.pos_end_effector = self.end_effector.get_position(relative_to=self.target)
        # self.or_target = self.target.get_orientation()
        # self.or_end_effector = self.end_effector.get_orientation(relative_to=self.target)
        #
        # self.dist_target = sqrt((self.pos_target[0])**2 + (self.pos_target[1])**2)
        # self.dist_end_effector = sqrt((self.pos_end_effector[0])**2 + (self.pos_end_effector[1])**2)

        if dist_ee_target < 0.1:
        # self.dist_target == self.dist_end_effector and self.or_target == self.or_end_effector:
            # +0.125>self.dist_end_effector>-0.125 and +2>self.or_end_effector>-2
            self.reached = 1
            reward = 100
            print('Target reached')
        else:
            reward = exp(-1.5*dist_ee_target)
            # reward = -1

        # obs = self.camera.capture_rgb()

        return reward


    def end_effector_pos(self):
        return self.end_effector.get_position()

    def target_position(self):
        return self.target.get_position()

    def get_joints_pos(self):
        self.joint1_pos = self.joint1.get_joint_position()
        self.joint2_pos = self.joint2.get_joint_position()
        return [self.joint1_pos,self.joint2_pos]

    def reset_target_position(self,random_=False,x=0.5,y=0.5):
        if random_ == True:
            xy_min = 0.3
            xy_max = 1.22
            x = random.random()*(xy_max-xy_min) + xy_min
            y_max = sqrt(1.22**2-x**2)
            y_min = 0
            y = random.random()*(y_max-y_min) + y_min

        self.target.set_position([x,y,0.125])
        self.pr.step()
        self.target_pos = self.target_position()

    def reset_robot_position(self,random_=False,joint1_pos=0,joint2_pos=-0.6109):
        if random_ == True:
            joint1_pos = random.random()*2*pi
            joint2_pos = random.random()*2*pi

        self.joint1.set_joint_position(joint1_pos,allow_force_mode=False) # radians
        self.joint2.set_joint_position(joint2_pos,allow_force_mode=False)
        self.pr.step()

    def display(self):
        img = self.camera.capture_rgb()
        plt.imshow(img,interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.pause(0.01)

    def random_agent(self,episodes=10):
        steps = 0
        steps_all = []
        for _ in range(episodes):
            while True:
                action = random.randrange(9)
                reward = self.step_(action)
                steps += 1

                if reward == 1:
                    steps_all.append(steps)
                    break

        return sum(steps_all)/episodes

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()
