from reacher_dqn import DQN
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from environment import environment
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

class results(object):

    def get_data_exp(self,exp_dir,label):
        self.n_exp = 1
        self.label = label
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path,'data/%s/log.txt'%exp_dir)
        data = pd.read_csv(data_path, sep='\t', header='infer')
        data = data.assign(exp="exp1")
        return data

    def get_data_multi_exp(self,exp_dir_all,label):
        # exp_dir_all shoud be a list with directory name
        self.n_exp = len(exp_dir_all)
        self.label = label
        exp_all = []
        for exp_id in range(self.n_exp):
            data_exp = self.get_data_exp(exp_dir_all[exp_id])
            data_exp = data_exp.assign(exp="exp%s"%(exp_id+1))
            exp_all.append(data_exp)

        data = pd.concat(exp_all)
        return data


    def plot_data(self,data,y_axis = "Averaged Return Training", x_axis="Step"):
        df = pd.DataFrame(data,columns=[x_axis,y_axis,'exp'])
        hue = None
        for col in data.columns:
            if col == 'exp':
                hue = "exp"

        g = sns.relplot(x = x_axis,y = y_axis, hue=hue, kind='line', data=df)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    def plot_return(self,data):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)

        with sns.axes_style("darkgrid"):
            for i in range(self.n_exp):
                idx = data.index[data['exp']=="exp%s"%str(i+1)]
                training_steps = np.array(data.loc[idx]["Step"].values, dtype=np.float64)
                mean = np.array(data.loc[idx]["Validation return"].values, dtype=np.float64)
                std = np.array(data.loc[idx]["Validation return std"].values, dtype=np.float64)
                ax.plot(training_steps, mean, label=self.label[i] ,c=clrs[i])
                ax.fill_between(training_steps, mean-std, mean+std, alpha=0.3, facecolor=clrs[i])

        plt.xlabel('Step')
        plt.ylabel('Validation return')
        ax.legend()


    def plot_all(self,data):
        self.plot_return(data)
        self.plot_steps(data)
        y_axis_ = ["Q-value evaluation", "Q-network loss", "Value-network loss", "Policy-network loss", "Alpha loss", "Log Pi"]
        for ys in y_axis_:
            self.plot_data(data, y_axis=ys, x_axis="Step")
        plt.show()

    def plot_steps(self, data):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)

        with sns.axes_style("darkgrid"):
            for i in range(self.n_exp):
                idx = data.index[data['exp']=="exp%s"%str(i+1)]
                training_steps = np.array(data.loc[idx]["Step"].values, dtype=np.float64)
                mean = np.array(data.loc[idx]["Validation steps"].values, dtype=np.float64)
                std = np.array(data.loc[idx]["Validation steps std"].values, dtype=np.float64)
                ax.plot(training_steps, mean, label=self.label[i] ,c=clrs[i])
                ax.fill_between(training_steps, mean-std, mean+std, alpha=0.3, facecolor=clrs[i])
        plt.xlabel('Step')
        plt.ylabel('Validation steps')
        ax.legend()

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
