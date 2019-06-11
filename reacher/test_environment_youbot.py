from env_youbot import environment
import random
from images_to_video import im_to_vid
import os
import time
from math import pi
import numpy as np

if not(os.path.exists('test_data')):
    os.makedirs('test_data')

vid = im_to_vid('test_data')
env = environment(rpa=1)
time.sleep(0.1)
steps = 80
num_joints = 2
velocity_test = 240*pi/180
img_ = []
env.reset()
error = False
for i in range(steps):
    prev_base_pos = np.array(env.base_ref.get_position())
    env.step([velocity_test, 0, 0])
    #change_joints_pos = np.abs(np.abs(prev_joints_pos) - np.abs(np.array(env.agent.get_joint_positions())))
    #desired_change_joints_pos = np.array(90*pi/180 * 0.05) # simulation time step of 50ms
    img_.append(env.render())

    #diff_joints_pos = (change_joints_pos - desired_change_joints_pos).sum()
    #if diff_joints_pos > 0.01 * velocity_test:
    #     print(prev_joints_pos)
    #     print(env.agent.get_joint_positions())
    #     error = True
    #     print("Error in position for given target velocity",diff_joints_pos,env.agent.get_joint_velocities())
    # else:
    #     print("Correct position from target velocity")
    #
    # if error == True:
    #     frame_error = (img_[-1])
    #     frame_error[0:40,0:40] = 0
    #     img_[-1] = frame_error

    if i%40 == 0 and i != 0:
        env.reset()
        obs = env.get_observation()
        # if obs[6] > 0.001 or obs[7] > 0.001:
        #     print("Error velocity not zero after reset")

# desired_target_pos = [0.4,0.4]
# desired_joints_pos = [0,0]
# env.reset_target_position(random_=False,x=desired_target_pos[0],y=desired_target_pos[1])
# env.reset_robot_position(random_=0, joints_pos=desired_joints_pos)

# target_pos = env.target_position()
# diff_target_pos = (np.array(desired_target_pos) - np.array(target_pos[0:2])).sum()
#
# if diff_target_pos > 0.05 * np.array(desired_target_pos).sum():
#     print('Error target position after reset')
# else:
#     print('Correct target position after reset')
#
# diff_joints_pos = (np.array(env.agent.get_joint_positions()) - np.array(desired_joints_pos)).sum()
# 
# if diff_joints_pos > 0.001:
#     print('Error joint position after reset')
# else:
#     print('Correct joint position after reset')

home = os.path.expanduser('~')
log_video_dir = os.path.join(home,'robotics_drl/reacher/','test_data/','episodes_video')
os.chdir(log_video_dir)
vid.from_list(img_,2)
env.terminate()
