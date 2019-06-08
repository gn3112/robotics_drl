from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms import Panda
import numpy as np
import math
import time
import random

LOOPS = 100
SCENE_FILE = join(dirname(abspath(__file__)), 'reacher_v2.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()
#agent = pr.get_arm(Panda)

# We could have made this target in the scene, but lets create one dynamically
#target = pr.create_primitive(type=pr.primitive_type_sphere,
#                             size=[0.05, 0.05, 0.05],
#                             color=[1.0, 0.1, 0.1],
#                             static=True, respondable=False)

position_min, position_max = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
#starting_joint_positions = agent.get_joint_positions()
joint1 = pr.get_joint('link_1')
joint1.set_control_loop_enabled(0)

for i in range(LOOPS):

    # Reset the arm at the start of each 'episode'
    #agent.set_joint_positions(starting_joint_positions)

    # Get a random position within a cuboid and set the target position
#    pos = np.random.uniform(position_min, position_max).tolist()
#    target.set_position(pos)

    # Get a path to the target (rotate so z points down)
#    path = agent.get_path(position=pos, orientation=[0, math.radians(180), 0])

#    if path is None:
#        print('Could not find path')
#        continue

    # Step the simulation and advance the agent along the path
#    done = False
#    for _ in range(10):
    joint1.set_joint_target_velocity(3*random.random())
    #agent.set_joint_target_velocity(3*random.random())
    pr.step()
        # We put a sleep purely for demo purposes
        # (otherwise it would finish the example too quickly!)
        #time.sleep(0.02)
    
    print(joint1.get_joint_position())
#   print('Reached target %d!' % i)

pr.stop()
pr.shutdown()


