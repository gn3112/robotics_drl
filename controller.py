from math import sqrt, radians, atan2, pi
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from youBot_all import youBotAll
from robot_all import robotAll
from youBot_base import youBotBase
from youBot_arm import youBotArm
import logz
import os
import numpy as np
import time
import cv2
import shutil
from math import sqrt, radians
import torchvision

class youBot_controller(robotAll):
    def __init__(self, OBS_LOW, ARM, BASE, REWARD, BOUNDARY, NAME):
        super().__init__('robot_scene.ttt', obs_lowdim=OBS_LOW, rpa=1, reward_dense=REWARD, demonstration_mode=True)

        self.OBS_LOW = OBS_LOW
        self.ARM = ARM
        self.BASE = BASE

        home = os.path.expanduser('~')
        self.logdir = os.path.join(home,'robotics_drl/data/demonstrations',NAME)
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)

        os.makedirs(self.logdir)

        if not self.OBS_LOW:
            os.makedirs(os.path.join(self.logdir,'image_observations'))

        os.chdir(self.logdir)
        self.log_file = open("log.txt","w")
        header = ["obs_%s"%(i) for i in range(self.observation_space()[0])] + ["next_obs_%s"%(j) for j in range(self.observation_space()[0])] + ["action_%s"%(k) for k in range(self.action_space)] + ["reward","done","steps","episode"]
        self.log_file.write('\t'.join(header))
        self.log_file.write('\n')

        if not OBS_LOW:
            if not(os.path.exists(os.path.join(self.logdir,'image_observations'))):
                os.makedirs(os.path.join(self.logdir,'image_observations'))

        self.arm.set_control_loop_enabled(1)

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

    def log_obs(self,L,imgs=None,ep=None,step=None):
        os.chdir(self.logdir)
        for i in range(len(L)):
            L[i] = str(L[i])
        self.log_file.write('\t'.join(L))
        self.log_file.write('\n')

        if not self.OBS_LOW:
            os.chdir(os.path.join(self.logdir,'image_observations'))
            ep = str(ep)
            # if len(ep) < 2:
            #     ep = '0' + ep

            step = str(step)
            # for _ in range(3 - len(step)):
            #     step = '0' + step

            img_ = imgs.view(-1,128,128)
            torchvision.utils.save_image(img_[:3,:,:], "episode%s_step%s_rgb.png" %(ep,step), normalize=True)
            torchvision.utils.save_image(img_[-1,:,:], "episode%s_step%s_d.png" %(ep,step), normalize=False)

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

    table = Shape('table')

    for ep in range(args.N_DEM):
        table.set_collidable(0)
        table.set_respondable(0)
        obs = controller.reset()
        time.sleep(0.1)
        steps = 0
        done = False
        path_base_done = False

        # Compute paths
        target_pos = controller.target_base.get_position()
        target_or = controller.target_base.get_orientation()[-1]
        # if not args.BASE:
        if args.BASE:
            try:
                path_base = controller.mobile_base.get_linear_path(target_pos,target_or)
            except:
                continue

            if path_base == None:
                print("No path could be computed")
                continue

            while True:
                action_base, path_base_done = path_base.step()
                steps += 1
                if args.ARM:
                    action_arm = [0,0,0,0]
                    action = np.concatenate((action_base,action_arm),axis=0)
                else:
                    action = action_base

                next_obs, reward, done = controller.step(action)
                if args.OBS_LOW:
                    next_obs = next_obs.tolist()
                    controller.log_obs(obs.tolist() + next_obs + controller.action + [reward,done,steps,ep+1])
                else:
                    next_obs_low = next_obs['low'].tolist()
                    controller.log_obs(obs['low'].tolist() + next_obs_low + controller.action + [reward,done,steps,ep+1], imgs=obs['high'],ep=ep+1,step=steps)

                obs = next_obs
                target_rel_pos = controller.target_base.get_position(relative_to=controller.mobile_base)
                dist_ee_target = sqrt(target_rel_pos[0]**2 + target_rel_pos[1]**2) - 0.25
                if not args.ARM:
                    if done:
                        break
                else:
                    if dist_ee_target < 0.1:
                        path_base_done=True
                        break

                if steps > 350:
                    print('Too many steps, start new episode')
                    args.N_DEM += 1
                    break

        if args.ARM:
            # path_base_done = True

            if path_base_done or not args.BASE:
                if args.BASE:
                    controller.mobile_base.set_motor_locked_at_zero_velocity(1)
                    controller.mobile_base.set_joint_target_velocities([0,0,0,0])
                    table.set_collidable(1)
                    table.set_respondable(1)

                try:
                    path_arm = controller.arm.get_nonlinear_path(position=target_pos,euler=[0,0,target_or])
                except:
                    args.N_DEM += 1
                    continue

                if path_arm == None:
                    print("No path for the arm could be computed")
                    continue

                #path_arm.visualize()
                time.sleep(0.1)
                path_arm_done = False
                done = False

                try:
                    while not done:
                        action_arm, path_arm_done = [0,0,0,0], path_arm.step() #Action zero as computed in the environment class
                        action_base = [0,0,0] if args.BASE else []
                        steps += 1
                        next_obs, reward, done = controller.step(np.concatenate((action_base,action_arm),axis=0).tolist())
                        if args.OBS_LOW:
                            next_obs = next_obs.tolist()
                            controller.log_obs(obs + next_obs + controller.action + [reward,done,steps,ep+1])
                        else:
                            next_obs_low = next_obs['low'].tolist()
                            controller.log_obs(obs['low'].tolist() + next_obs_low + controller.action + [reward,done,steps,ep+1], imgs=obs['high'],ep=ep+1,step=steps)
                        obs = next_obs
                except:
                    args.N_DEM += 1
                    continue

                #path_arm.clear_visualization()

        if ep % 10 == 0 and ep != 0:
            print("Generated 10 episodes")

    controller.log_file.close()

    controller.terminate()
if __name__ == "__main__":
    main()
