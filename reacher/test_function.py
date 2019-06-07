from env_reacher_v2 import environment
import random
from images_to_video import im_to_vid
import os

if not(os.path.exists('test_data')):
    os.makedirs('test_data')

vid = im_to_vid('test_data')
env = environment(continuous_control=True)
steps = 400
img_ = []
env.reset()
for i in range(steps):
    action = [(3 * random.random() - 1.5),(3 * random.random() - 1.5)]
    env.step([1.5,1.5])
    img_.append(env.render())

    if i%500 == 0:
        env.reset()
        print(env.get_obs())

home = os.path.expanduser('~')
log_video_dir = os.path.join(home,'robotics_drl/reacher/','test_data/','episodes_video')
os.chdir(log_video_dir)
vid.from_list(img_,1)
env.terminate()
