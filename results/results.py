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
        self.label = [label]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        home_dir = os.path.expanduser('~')
        data_path = os.path.join(home_dir,'robotics_drl/data/%s/log.txt'%exp_dir)
        data = pd.read_csv(data_path, sep='\t', header='infer')
        # data['Policy-network loss'] = data['Policy-network loss'].str.strip('[]')
        data['Alpha'] = data['Alpha'].str.strip('[]')
        data = data.astype(float)
        data = data.assign(exp=label)
        return data

    def get_data_multi_exp(self,exp_dir_all,label):
        # exp_dir_all shoud be a list with directory name
        self.n_exp = len(exp_dir_all)
        self.label = label
        exp_all = []
        for exp_id in range(self.n_exp):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            home_dir = os.path.expanduser('~')
            data_path = os.path.join(home_dir,'robotics_drl/data/%s/log.txt'%exp_dir_all[exp_id])
            data_exp = pd.read_csv(data_path, sep='\t', header='infer')
            data_exp = data_exp.assign(exp=label[exp_id])
            exp_all.append(data_exp)
        data = pd.concat(exp_all)
        return data


    def plot_data(self,data,y_axis = "Averaged Return Training", x_axis="Training steps"):
        df = pd.DataFrame(data,columns=[x_axis,y_axis,'exp'])
        hue = None
        for col in data.columns:
            if col == 'exp':
                hue = "exp"

        g = sns.relplot(x = x_axis,y = y_axis, hue=hue, kind='line', data=df)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    def plot_return(self,data_all):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)

        with sns.axes_style("darkgrid"):
            for i in range(self.n_exp):
                data = data_all[data_all['exp'].str.contains(self.label[i])]
                training_steps = np.array(data["Training steps"].values, dtype=float)
                mean = data["Validation return"].values
                mean_min = np.min(mean)
                mean_max = np.max(mean)
                mean = (mean - mean_min)/(mean_max - mean_min)
                std = np.array(data["Validation return std"].values, dtype=float)
                ax.plot(training_steps, mean, label=self.label[i] ,c=clrs[i])
                ax.fill_between(training_steps, mean-std, mean+std, alpha=0.3, facecolor=clrs[i])

        plt.xlabel('Training steps', fontsize=30)
        plt.ylabel('Validation return')
        ax.legend()

    def plot_all(self,data_all):
        for i in range(self.n_exp):
            idx = data_all.index[data_all['exp']==self.label[i]]
            data = data_all.loc[idx]
            self.plot_return(data)
            self.plot_steps(data)
            y_axis_ = ["Q-value evaluation", "Q-network loss", "Cumulative Success", "Value-network loss", "Policy-network loss", "Alpha loss", "Log Pi"]
            for ys in y_axis_:
                self.plot_data(data, y_axis=ys, x_axis="Training steps")
            plt.show()

    def plot_steps(self, data_all):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)
        for i in range(self.n_exp):
            with sns.axes_style("darkgrid"):
                data = data_all[data_all['exp'].str.contains(self.label[i])]
                training_steps = np.array(data["Training steps"].values, dtype=float)
                mean = np.array(data["Validation steps"].values, dtype=float)
                # mean = 1 / (mean / np.linalg.norm(mean))
                std = np.array(data["Validation steps std"].values, dtype=float)
                # std = 1 / (std / np.linalg.norm(std))
                ax.plot(training_steps, mean, label=self.label[i] ,c=clrs[i])
                ax.fill_between(training_steps, mean-std, mean+std, alpha=0.3, facecolor=clrs[i])
        ax.plot([0,500000],[60,60],'--',label='threshold task failure')
        plt.xlabel('Training steps', fontsize=14)
        plt.ylabel('Validation score', fontsize=14)
        ax.legend()

    def sample_episode(exp_dir):
        env = environment()
        env.reset_target_position(random_=False)
