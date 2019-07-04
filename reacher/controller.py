from math import sqrt, radians, atan2, pi
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from env_youbot import environment
import logz
import os
import numpy as np
import time
import cv2

class youBot_controller(environment):
    def __init__(self, OBS_LOW, ARM, BASE, NAME):
        super().__init__(manipulator=False, base=True, obs_lowdim=True, rpa=1, demonstration_mode=True)

        self.OBS_LOW = OBS_LOW
        self.ARM = ARM
        self.BASE = BASE

        home = os.path.expanduser('~')
        self.logdir = os.path.join(home,'robotics_drl/reacher/data/demonstrations',NAME)
        if not(os.path.exists(self.logdir)):
            os.makedirs(self.logdir)

        os.chdir(self.logdir)
        self.log_file = open("log.txt","w")
        header = ["obs_%s"%(i) for i in range(self.observation_space()[0])] + ["next_obs_%s"%(j) for j in range(self.observation_space()[0])] + ["reward","done","steps","episode"]
        self.log_file.write('\t'.join(header))
        self.log_file.write('\n')

        if not OBS_LOW:
            if not(os.path.exists(os.path.join(self.logdir,'image_observations'))):
                os.makedirs(os.path.join(self.logdir,'image_observations'))

    def is_base_reached(self):
        pos_v = self.intermediate_target.get_position(relative_to=self.base_ref)
        if sqrt(pos_v[0]**2 + pos_v[1]**2) < 0.1:
            return True

    def generate_trajectories(self):
        # Reaching with base and then with manipulator
        base_status = self.generate_base_trajectories()
        if base_status == False:
            print("Base reached first target!")
            done = self.generate_arm_trajectories()
            return done
        else:
            return False

    def follow_path(self,path):
        done = False
        while not done:
            done = path.step()
            self.pr.step()
        return done

    def log_obs(self,L,ep=None,step=None):
        if self.OBS_LOW:
            os.chdir(self.logdir)
            for i in range(len(L)):
                L[i] = str(L[i])
            self.log_file.write('\t'.join(L))
            self.log_file.write('\n')
        else:
            os.chdir(os.path.join(self.logdir,'image_observations'))
            self.cv2.imwrite("episode%s_step%s.png" %(ep,step), img)

    def generate_arm_trajectories(self):
        path = self.get_arm_path()

        if path is None:
            print("NO PATH")
            return False

        done = self.follow_path(path)

        return done

    def store_transitions(self):
        img = self.camera.capture_rgb()*256
        self.img_all.append(img)

    def get_arm_path(self):
        path = self.arm.get_path(position=self.target.get_position(), euler=[0, radians(180), 0])
        return path

    def generate_transitions(self, ep, obs_low=True, reward=False):
        self.reset()
        for _ in range(ep):
            while True:
                _, done = self.get_action_to_target()
                if done:
                    self.reset()
                    break

def main():
    DEM_N = 500
    OBS_LOW = True
    ARM = False
    BASE = True
    NAME = "youbot_navig_low_demonstrations"

    if not(os.path.exists('data/demonstrations')):
        os.makedirs('data/demonstrations')

    controller = youBot_controller(OBS_LOW, ARM, BASE, NAME)
    controller.reset()

    for ep in range(DEM_N):
        obs = controller.reset().tolist()
        steps = 0
        done = False
        while not done:
            target_pos = controller.target_base.get_position()[:2]
            target_or = controller.target_base.get_orientation()[0]
            path = controller.mobile_base.set_linear_path([target_pos[0],target_pos[1],target_or])
            action, path_done = path.step_path()
            steps += 1
            next_obs, reward, done = controller.step(action)

            controller.log_obs(obs.tolist() + next_obs.tolist() + [reward,done,steps,ep+1])

            obs = next_obs

    controller.log_file.close()

    controller.terminate()
if __name__ == "__main__":
    main()
