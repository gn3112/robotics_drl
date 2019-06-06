from pyrep import PyRep
from math import pi
from os.path import dirname, join, abspath
import torch

class youbot_base(object):
    def __init__(self):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), 'youbot.ttt')
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        self.target = self.pr.get_object('target')
        self.base_ref = self.pr.get_dummy('youbot_ref')
        self.youBot = self.pr.get_object('youBot')
        #self.camera = self.pr.get_vision_sensor('Vision_sensor')

        self.wheel_joint_handle = []
        joint_name = ['rollingJoint_fl','rollingJoint_rl','rollingJoint_rr','rollingJoint_fr']
        for joint in joint_name:
            self.wheel_joint_handle.append(self.pr.get_joint(joint))

        ForwBackVel_range = [-240,240]
        LeftRightVel_range = [-240,240]
        RotVel_range = [-240,240]
        self.move_base() #Set velocity to 0

        self.done = False
        self.xy_vel = []

    def move_base(self,forwBackVel=0,leftRightVel=0,rotVel=0):
        self.wheel_joint_handle[0].set_joint_target_velocity(-forwBackVel-leftRightVel-rotVel)
        self.wheel_joint_handle[1].set_joint_target_velocity(-forwBackVel+leftRightVel-rotVel)
        self.wheel_joint_handle[2].set_joint_target_velocity(-forwBackVel-leftRightVel+rotVel)
        self.wheel_joint_handle[3].set_joint_target_velocity(-forwBackVel+leftRightVel+rotVel)
        for rpa in range(5):
            pos_prev = self.base_ref.get_position()
            self.pr.step()
            pos_next = self.base_ref.get_position()

            for i in range(2):
                self.xy_vel[i] = (pos_next[i] - pos_prev[i]) / 0.05

    def render(self):
        img = self.camera.capture_rgb()
        return img*256

    def get_obs(self):
        obs = []
        or = self.base_ref.get_orientation()
        pos = self.base_ref.get_position()

        vec_to_target = self.target_base.get_position() - self.base_ref.get_position()
        return obs.append(or, pos, self.xy_vel, vec_to_target)

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

        state = self.get_obs()
        return torch.tensor(state, dtype=torch.float32) ,reward, self.done

    def reset(self,, ):

        reset_target_position(self,randT=True)

        state = self.get_obs()
        return torch.tensor(state, dtype=torch.float32)

    def reset_target_position(self,randT=False, position=[0,0]):
        if randT:
            x_T,y_T = self.rand_bound()
        else:
            x_T,y_T = position

        self.target.set_position([x_T,y_T,0.0275])

    def reset_robot_position(self,position=[0.5,0.5], orientation=0):
        x_L, y_L = position
        self.youBot.set_position([x_L,y_L,0.095750])
        self.youBot.set_orientation([orientation,90,180])

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()

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
