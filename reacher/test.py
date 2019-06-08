from pyrep import PyRep
from os.path import dirname, join, abspath
import time

panda = False

pr = PyRep()
SCENE_FILE = join(dirname(abspath(__file__)), 'panda_scene.ttt' if panda==True else 'reacher_v2.ttt')
# Launch the application with a scene file that contains a robot
pr.launch(SCENE_FILE,headless=True) 
pr.set_simulation_timestep(0.05)
pr.start()  # Start the simulation

joint = pr.get_joint('joint_1' if panda==True else 'joint_1')
joint.set_control_loop_enabled(0)
joint.set_motor_locked_at_zero_velocity(1)
print('Joint Pos:',joint.get_joint_position())
print('Joint Type:',joint.get_joint_type())
print('Joint Mode',joint.get_joint_mode())
print('Control loop enable:',joint.is_control_loop_enabled())
print('Motor enable:',joint.is_motor_enabled())
print('Motor locked at 0 velocity:',joint.is_motor_locked_at_zero_velocity())

#joint.set_joint_mode(5)

for _ in range(4):
    joint.set_joint_target_velocity(1)
    pr.step()
    print('Joint Pos:',joint.get_joint_position())
    print('Joint Targ Vel:',joint.get_joint_target_velocity())
    print('Joint Vel:',joint.get_joint_velocity())
pr.stop()
pr.shutdown()
