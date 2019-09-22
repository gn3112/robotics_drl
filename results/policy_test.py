import torch
from youBot_all import youBotAll
from models import SoftActorGated
import os
import numpy as np
from math import sqrt
from pyrep.objects.vision_sensor import VisionSensor
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
def resize(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((512,512)),
                        T.ToTensor()])
    return resize(np.uint8(a*255))

env = youBotAll('youbot_navig2.ttt', obs_lowdim=True, rpa=3, reward_dense=False, boundary=1)
actor = SoftActorGated(256, env.action_space, env.observation_space(), flow_type='tanh', flows=0).cuda()
camera_side = VisionSensor('side_camera')
path_model = os.path.join(os.path.expanduser('~'), 'robotics_drl/data/youbot_all_final_21-08-2019_22-32-00/model.pkl')
actor.load_state_dict(torch.load(path_model))

actor.eval()
state = env.reset()
for i in range(2):
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        with torch.no_grad():
            # state['low'] = state['low'].to(device)
            # state['high'] = state['high'].to(device)
            img = resize(camera_side.capture_rgb())
            save_image(img,'y_%s.png'%steps)
            action, _, gate  = actor(state.cuda(), log_prob=False, deterministic=True)
            print(sqrt(np.sum(np.array(env.target_base.get_position(relative_to=env.tip))**2)), gate)
        state, reward, done = env.step(action.squeeze().cpu().numpy())
        if reward == env.reward_termination:
            print('Task %s completed'%(i))
            break

        if steps > 250:
            break


env.terminate()
#
# prev = Image.open('y_%s.png'%(1))
# prev = prev.convert('RGBA')
# for i in range(2,steps,4):
#     overlay = Image.open('y_%s.png'%(i))
#
#     overlay = overlay.convert('RGBA')
#     print(i)
#     prev = Image.blend(prev, overlay, 0.9)
#
# prev.save('overlay1.png')
