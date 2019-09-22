import torch
from youBot_all import youBotAll
from models import SoftActorGated, SoftActorFork
import os
import numpy as np
from math import sqrt
from pyrep.objects.vision_sensor import VisionSensor
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def resize(a):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((512,512)),
                        T.ToTensor()])
    return resize(np.uint8(a*255))
device = torch.device('cuda')

env = youBotAll('youbot_navig2.ttt', obs_lowdim=True, rpa=3, reward_dense=False, boundary=1)
actor = SoftActorGated(256, env.action_space, env.observation_space(), flow_type='tanh', flows=0)
actor2 = SoftActorFork(256, env.action_space, env.observation_space(), flow_type='tanh', flows=0)
camera_side = VisionSensor('side_camera')
path_model = os.path.join(os.path.expanduser('~'), 'robotics_drl/data/youbot_all_final_21-08-2019_22-32-00/model.pkl')
actor.load_state_dict(torch.load(path_model))
actor2.load_state_dict(torch.load(os.path.join('/media/georges/disk/youbot_all_fork_OK/model.pkl')))

actor.to(device)
actor.eval()
actor2.to(device)
actor2.eval()
distance_all = []
arm_all = []
base_all = []
steps_all_all = []
gate_all = []
for i in range(2):
    while True:
        state = env.reset()
        if sqrt(np.sum(np.array(env.target_base.get_position(relative_to=env.tip))**2)) > 0.9:
            break

    steps = 0
    distance = []
    arm = []
    base = []
    steps_all = []
    while True:
        steps += 1
        with torch.no_grad():
            img = resize(camera_side.capture_rgb())
            if i > 0:
                action, _ = actor2(state.to(device), log_prob=False, deterministic=True)
            else:
                action, _, gate  = actor(state.to(device), log_prob=False, deterministic=True)
                gate_all.append(gate.squeeze().cpu().tolist())

            print(sqrt(np.sum(np.array(env.target_base.get_position(relative_to=env.tip))**2)), gate)

        arm.append(np.abs(np.mean(np.array(action.squeeze().cpu()[3:]))))
        base.append(np.abs(np.mean(np.array(action.squeeze().cpu()[:3]))))
        steps_all.append(steps)
        distance.append(sqrt(np.sum(np.array(env.target_base.get_position(relative_to=env.tip))**2)))
        state, reward, done = env.step(action.squeeze().cpu().numpy())

        if steps > 100 or reward == env.reward_termination:
            distance_all.append(distance)
            arm_all.append(arm)
            base_all.append(base)
            steps_all_all.append(steps_all)
            if reward == env.reward_termination:
                print('Task %s completed'%(i))
            else:
                print('Too many steps %s'%i)
            break


env.terminate()
# distance_all = np.mean(np.array(distance_all), axis=1)
# arm_all = np.mean(np.array(arm_all), axis=0)
# base_all = np.mean(np.array(base_all), axis=0)
f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True)
ax1.plot(steps_all_all[0], distance_all[0], 'b', label='gate', linewidth=1.5)
ax1.plot(steps_all_all[1], distance_all[1], 'g', label='fork', linewidth=1.5)
ax1.title.set_text('(a) Distance to target')
ax2.plot(steps_all_all[0], arm_all[0], 'b', label='gate', linewidth=1.5)
ax2.plot(steps_all_all[1], arm_all[1], 'g', label='fork', linewidth=1.5)
ax2.title.set_text('(b) Magnitude arm action')
ax3.plot(steps_all_all[0], base_all[0], 'b', label='gate', linewidth=1.5)
ax3.plot(steps_all_all[1], base_all[1], 'g', label='fork', linewidth=1.5)
ax3.title.set_text('(c) Magnitude base action')
print(steps_all_all[0], gate_all)
ax4.plot(steps_all_all[0], gate_all, 'b', label='gate', linewidth=1.5)
ax4.title.set_text('(d) Gate value')
ax1.legend(loc="upper right")
plt.show()
