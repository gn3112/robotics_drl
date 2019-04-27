from reacher_dqn import DQN
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from environment import environment
import torch
from torchvision import transforms as T
from PIL import Image

class results(object):
    
    def get_data_exp(exp_dir: str):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path,'data/%s/log.txt'%exp_dir)
        data = pd.read_csv(data_path, sep='\t', header='infer')
        return data

    def get_data_multi_exp(exp_dir_all): # exp_dir_all shoud be a list with directory name
        exp_all = []
        for exp_id in range(len(exp_dir_all)):
            data_exp = get_data_exp(exp_dir_all[exp_id])
            data_exp.assign(exp='exp%s'%exp_id)
            exp_all.append(data_exp)

        data = pd.concat(exp_all)
        return data

    def plot_data(data,y_axis = "Averaged Rewards", x_axis="Number of episodes"):
        df = pd.DataFrame(data,columns=[x_axis,y_axis])
        hue = None
        for col in data.columns:
            if col == 'exp':
                hue = 'exp'

        g = sns.relplot(x = x_axis,y = y_axis, hue=hue, kind='line', data=df)
        plt.show()

    def sample_episode(exp_dir):
        env = environment()
        env.reset_target_position(random_=False)
        env.reset_robot_position()

        resize = T.Compose([T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),
                            T.Resize(64, interpolation=Image.BILINEAR),
                            T.ToTensor()])

        policy_net = DQN(64,64)
        policy_net.load_state_dict(torch.load(os.path.join(dir_path,
                                              'data/%s/model.pkl'%exp_dir),
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
