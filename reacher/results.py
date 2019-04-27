from reacher_dqn import DQN
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from environment import environment
import torch
from torchvision import transforms as T
from PIL import Image

def get_data(path: str):
    data = pd.read_csv(path, sep='\t', header='infer')
    return data

def plot_data():
    return None

#/data/exp3_09-04-2019_20-28-38
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,'data/exp4_13-04-2019_19-51-33/log.txt')
data = get_data(data_path)
idx_column = data.columns
df = pd.DataFrame(data,columns=['Gradient update','Averaged Rewards'])
df2 = pd.DataFrame(data,columns=['Gradient update','Average q-value evaluation'])
df3 = pd.DataFrame(data,columns=['Gradient update','Epsilon threshold'])

# sns.set('darkgrid', font_scale=2)
print(idx_column)
g = sns.relplot(x = 'Gradient update',y = 'Averaged Rewards', kind='line', data=df)
h = sns.relplot(x = 'Gradient update',y = 'Average q-value evaluation', kind='line', data=df2)
g.fig.autofmt_xdate()
h.fig.autofmt_xdate()
plt.show()

env = environment()
env.reset_target_position(random_=False)
env.reset_robot_position()

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(64, interpolation=Image.BILINEAR),
                    T.ToTensor()])

policy_net = DQN(64,64)
policy_net.load_state_dict(torch.load(os.path.join(dir_path,
                                      'data/exp4_13-04-2019_19-51-33/model.pkl'),
                                      map_location='cpu'))
policy_net.eval()
steps = 0
while True:
    img = resize(env.render()).unsqueeze(0)
    action = policy_net(img).argmax(1).view(1,-1)
    r = env.step_(action)
    env.display()
    steps += 1
    if r == 100:
        print('Target Reached !!!')
        break
    elif steps == 150:
        steps = 0
        env.reset_target_position(random_=True)
        env.reset_robot_position(random_=True)
