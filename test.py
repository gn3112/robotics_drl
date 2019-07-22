from pyrep import PyRep
from pyrep.robots.arms import youBot
from os.path import dirname, join, abspath
import os
import time
import numpy as np
import math
from images_to_video import im_to_vid

if not(os.path.exists('test_data')):
    os.makedirs('test_data')
imtovid = im_to_vid('test_data')

pr = PyRep()
SCENE_FILE = join(dirname(abspath(__file__)), 'youbot.ttt')
# Launch the application with a scene file that contains a robot
pr.launch(SCENE_FILE,headless=True) 
pr.start()
time.sleep(0.1)
LOOPS = 10
agent = pr.get_arm(youBot)
target = pr.get_object('target')
camera = pr.get_vision_sensor('side_camera')
position_min, position_max = [-0.3, 0.5, 0.25], [0.3, 0.7, 0.4]
starting_joint_positions = agent.get_joint_positions()

img_all = []
for i in range(LOOPS):
    agent.set_joint_positions(starting_joint_positions)
    pos = np.random.uniform(position_min, position_max).tolist()
    target.set_position(pos)
    
    path = agent.get_path(position=pos, orientation=[0, math.radians(180), 0])

    if path is None:
        print('NO PATH')
        continue

    done = False
    img = camera.capture_rgb()*256
    while not done:
        img_all.append(img)
        done = path.step()
        time.sleep(0.02)
        img = camera.capture_rgb()*256

    print("Reached target %s"%i)

home = os.path.expanduser('~')
os.chdir(join(home,'robotics_drl/reacher'))
imtovid.from_list(img_all,0)

pr.stop()
pr.shutdown()
