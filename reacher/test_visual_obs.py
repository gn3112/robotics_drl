from pyrep import PyRep
from env_reacher_v2 import environment
from os.path import dirname, join, abspath
import os
import time
import numpy as np
import math
import torchvision
import torch
from torchvision import transforms as T

def change_dir():
    home = os.path.expanduser('~')
    os.chdir(join(home,'robotics_drl/reacher'))

def resize(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        T.ToTensor()])
    return resize(np.uint8(a))

env = environment(obs_lowdim=False)
time.sleep(0.1)
env.reset()
img = env.render()
print(np.shape(img))
img = resize(img).view(-1,64,64)
change_dir()
print(img.size())
torchvision.utils.save_image(img, "no_processing.png", normalize=True)
print(img.min(),img.max(),img.mean())

obs = env.reset().view(-1,64,64)
img = obs[0,:,:].view(-1,64,64)
change_dir()
torchvision.utils.save_image(img, "processing.png",normalize=True)
print(img.min(),img.max(),img.mean())


env.terminate()
