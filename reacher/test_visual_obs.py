from pyrep import PyRep
from env_reacher_v2 import environment
from os.path import dirname, join, abspath
import os
import time
import numpy as np
import math
import torchvision

def change_dir():
    home = os.path.expanduser('~')
    os.chdir(join(home,'robotics_drl/reacher'))

env = environment(obs_lowdim=False)
img = torch.tensor(env.render()/256).view(-1,64,64)
change_dir()
torchvision.utils.save_image(img, "no_processing.png")

obs = env.reset().view(-1,64,64)
img = obs[0,:,:].view(-1,64,64)
change_dir()
torchvision.utils.save_image(img, "processing.png")



env.terminate()
