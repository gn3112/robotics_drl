import torch
import numpy as np
from images_to_video import im_to_vid

class evaluation_sac(object):
    def __init__(self, env, expdir, device, n_states=15):
        self.states_eval = []
        self.env = env
        self.imtovid = im_to_vid(expdir)
        self.expdir = expdir
        self.device = device
        self.ep = 0

        for _ in range(n_states):
            state = self.env.reset()
            action = self.env.sample_action()
            self.states_eval.append([state,action])

    def get_qvalue(self,critic_1,critic_2):
        qvalues = 0
        for state_action in self.states_eval:
            with torch.no_grad():
                action = torch.tensor(state_action[1]).float().to(self.device).view(-1,self.env.action_space)
                if self.env.obs_lowdim:
                    state = state_action[0].float().to(self.device)
                    qvalues += (critic_1(state,action)[0] + critic_2(state,action)[0])/2
                else:
                    qvalues += (critic_1(state_action[0]['low'].float().to(self.device),action)[0] + critic_2(state_action[0]['low'].float().to(self.device),action)[0])/2

        return (qvalues/len(self.states_eval)).item()

    def sample_episode(self, actor, save_video=True, n_episodes=2):
        steps_all = []
        return_all = []
        with torch.no_grad():
            for ep in range(n_episodes):
                state, done, total_reward, steps_ep = self.env.reset(), False, 0, 0
                img_ep = []
                while True:
                    if self.env.obs_lowdim:
                        action, _, _ = actor(state.float().to(self.device), log_prob=False, deterministic=True)
                    else:
                        state['low'] = state['low'].float().to(self.device)
                        state['high'] = state['high'].float().to(self.device)
                        action, _, _ = actor(state, log_prob=False, deterministic=True)
                    steps_ep += 1
                    state, reward, done = self.env.step(action.detach().cpu().squeeze(dim=0).tolist())
                    total_reward += reward
                    img_ep.append(self.env.render())
                    if steps_ep > self.env.step_limit() - 1 or done==True:
                        if save_video==True: self.save_ep_video(img_ep)
                        steps_all.append(steps_ep)
                        return_all.append(total_reward)
                        break

        return np.array(return_all), np.array(steps_all)

    def save_ep_video(self,imgs):
        self.ep += 1
        self.imtovid.from_list(imgs,self.ep)

    def record_episode(self,img_all):
        logdir = os.path.expanduser('~') + '/robotics_drl/' + self.expdir + '/episode%s'%(self.ep)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        size = 64, 64
        for idx, img in enumerate(img_all):
            ndarr = img.reshape(64,64,3)
            im = Image.fromarray(ndarr)
            imdir = os.path.join(logdir,'step%s.jpg'%idx)
            im.resize(size, Image.BILINEAR)
            im.save(imdir,"JPEG")
        self.ep += 1

class evaluation_dqn(object):
    def __init__(self, env, expdir, n_states=20):
        self.states_eval = []
        self.env = env
        self.imtovid = im_to_vid(expdir)
        self.expdir = expdir
        self.ep = 0
        # self.resize = T.Compose([T.ToPILImage(),
        #                          T.Grayscale(num_output_channels=1),
        #                          T.Resize(64, interpolation=Image.BILINEAR),
        #                          T.ToTensor()])
        for _ in range(n_states):
            self.env.reset_robot_position(random_=True)
            self.env.reset_target_position(random_=False)
            obs = ((self.env.get_obs()))
            self.states_eval.append(obs)

    def get_qvalue(self,policy_net):
        policy_net.eval()
        qvalues = 0
        for _, obs in enumerate(self.states_eval):
            with torch.no_grad():
                qvalues += policy_net(torch.from_numpy(obs).view(1,-1)).max(1)[0]
        return (qvalues/len(self.states_eval))[0].item()

    def sample_episode(self,policy_net,save_video=False,n_episodes=5,threshold_ep=60):
        # 0.1 greedy policy or 100% action from network ?
        policy_net.eval()
        steps_all = []
        return_all = []
        for _ in range(n_episodes):
            steps = 0
            return_ = 0
            img_ep = deque([])
            self.env.reset_robot_position(random_=True)
            self.env.reset_target_position(random_=False)
            while True:
                obs = (self.env.get_obs())
                img = self.env.render()
                img_ep.append(img)
                with torch.no_grad():
                    action = policy_net(torch.from_numpy(obs).view(1,-1)).argmax(1).view(1,-1)
                reward, done = self.env.step_(action)
                steps += 1
                return_ += reward
                if done:
                    break
                elif steps == threshold_ep:
                    break

            if save_video==True: self.save_ep_video(img_ep)
            steps_all.append(steps)
            return_all.append(return_)
        return return_all, steps_all

    def save_ep_video(self,imgs):
        self.ep += 1
        self.imtovid.from_list(imgs,self.ep)

    def record_episode(self,img_all):
        logdir = os.path.expanduser('~') + '/robotics_drl/' + self.expdir + '/episode%s'%(self.ep)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        size = 64, 64
        for idx, img in enumerate(img_all):
            ndarr = img.reshape(64,64,3)
            im = Image.fromarray(ndarr)
            imdir = os.path.join(logdir,'step%s.jpg'%idx)
            im.resize(size, Image.BILINEAR)
            im.save(imdir,"JPEG")
        self.ep += 1
