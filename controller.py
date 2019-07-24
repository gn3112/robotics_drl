from math import sqrt, radians, atan2, pi
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from youBot_all import youBotAll
from youBot_base import youBotBase
from youBot_arm import youBotArm
import logz
import os
import numpy as np
import time
import cv2

class youBot_controller(youBotBase):
    def __init__(self, OBS_LOW, ARM, BASE, REWARD, BOUNDARY, NAME):
        super().__init__(obs_lowdim=OBS_LOW, rpa=1, reward_dense=REWARD, demonstration_mode=True)

        self.OBS_LOW = OBS_LOW
        self.ARM = ARM
        self.BASE = BASE

        home = os.path.expanduser('~')
        self.logdir = os.path.join(home,'robotics_drl/data/demonstrations',NAME)
        if not(os.path.exists(self.logdir)):
            os.makedirs(self.logdir)

        os.chdir(self.logdir)
        self.log_file = open("log.txt","w")
        header = ["obs_%s"%(i) for i in range(self.observation_space()[0])] + ["next_obs_%s"%(j) for j in range(self.observation_space()[0])] + ["action_%s"%(k) for k in range(self.action_space)] + ["reward","done","steps","episode"]
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



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', required=True, type=str)
    parser.add_argument('--OBS_LOW', action='store_true')
    parser.add_argument('--ARM' , action='store_true')
    parser.add_argument('--BASE' , action='store_true')
    parser.add_argument('--N_DEM', required=True, type=int)
    parser.add_argument('--REWARD_DENSE', action='store_true')
    parser.add_argument('--BOUNDARY', default=1, type=float)

    args = parser.parse_args()

    if not(os.path.exists('data/demonstrations')):
        os.makedirs('data/demonstrations')

    controller = youBot_controller(args.OBS_LOW, args.ARM, args.BASE, args.REWARD_DENSE, args.BOUNDARY, args.save_file)
    controller.reset()

    for ep in range(args.N_DEM):
        obs = controller.reset().tolist()
        time.sleep(0.1)
        steps = 0
        done = False
        path_base_done = False

        # Compute paths
        target_pos = controller.target_base.get_position()
        target_or = controller.target_base.get_orientation()[-1]
        if args.BASE:
            path_base = controller.mobile_base.get_linear_path(target_pos,target_or)

            if path_base == None:
                print("No path could be computed")
                continue

            while True:
                action_base, path_base_done = path_base.step()
                steps += 1
                if args.ARM:
                    action_arm = [0,0,0]
                    action = np.concatenate((action_base,action_arm),axis=0)
                else:
                    action = action_base
                next_obs, reward, done = controller.step(action)
                next_obs = next_obs.tolist()
                controller.log_obs(obs + next_obs + controller.action + [reward,done,steps,ep+1])

                obs = next_obs
                if not args.ARM:
                    if done:
                        break
                else:
                    if path_base_done:
                        break

                if steps > 350:
                    print('Too many steps, start new episode')
                    args.N_DEM += 1
                    break

        if args.ARM:
            if path_base_done or not args.BASE:
                if args.BASE:
                    controller.mobile_base.set_motor_locked_at_zero_velocity(1)
                    controller.mobile_base.set_joint_target_velocities([0,0,0,0])

                try:
                    path_arm = controller.arm.get_nonlinear_path(position=target_pos,euler=[0,0,target_or])
                except:
                    args.N_DEM += 1
                    continue

                if path_arm == None:
                    print("No path for the arm could be computed")
                    continue

                time.sleep(0.1)
                path_arm_done = False
                done = False
                while not done:
                    action_arm, path_arm_done = [0,0,0], path_arm.step() #Action zero as computed in the environment class
                    action_base = [0,0,0] if args.BASE else []
                    steps += 1
                    next_obs, reward, done = controller.step(np.concatenate((action_base,action_arm),axis=0))
                    next_obs = next_obs.tolist()
                    controller.log_obs(obs + next_obs + controller.action + [reward,done,steps,ep+1])

                    obs = next_obs

        if ep % 10 == 0 and ep != 0:
            print("Generated 10 episodes")

    controller.log_file.close()

    controller.terminate()
if __name__ == "__main__":
    main()
