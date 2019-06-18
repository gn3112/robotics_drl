from collections import deque
import random
import torch
from torch import optim
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from env_reacher_v2 import environment
from models import Critic, SoftActor, create_target_network, update_target_network
import logz
import inspect
import time
import os
import numpy as np
from drl_evaluation import evaluation_sac
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAC implementation from spinning-up-basic

def setup_logger(logdir,locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(params)

def optimise(args):
    return None

def train(BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START, ENV, OBSERVATION_LOW, VALUE_FNC, logdir):
    setup_logger(logdir, locals())
    ENV = __import__(ENV)
    env = ENV.environment(obs_lowdim=OBSERVATION_LOW)
    time.sleep(0.1)
    action_space = env.action_space()
    obs_space = env.observation_space()
    step_limit = env.step_limit()
    if OBSERVATION_LOW:
        def resize(a):
            return torch.tensor(a)
    else:
        def resize(a):
            # resize = T.Compose([T.ToPILImage(),
            #                     T.Grayscale(num_output_channels=1),
            #                     T.Resize(obs_space[0], interpolation=Image.BILINEAR),
            #                     T.ToTensor()])
            return a

    actor = SoftActor(HIDDEN_SIZE, action_space, obs_space).float().to(device)
    critic_1 = Critic(HIDDEN_SIZE, 1, obs_space, action_space, state_action= True).float().to(device)
    critic_2 = Critic(HIDDEN_SIZE, 1, obs_space, action_space, state_action= True).float().to(device)
    if VALUE_FNC:
        value_critic = Critic(HIDDEN_SIZE, 1, obs_space, action_space).float().to(device)
        target_value_critic = create_target_network(value_critic).float().to(device)
        value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
    else:
        target_critic_1 = create_target_network(critic_2)
        target_critic_2 = create_target_network(critic_2)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
    D = deque(maxlen=REPLAY_SIZE)

    eval_ = evaluation_sac(env,logdir,device,resize)

    #Automatic entropy tuning init
    target_entropy = -np.prod(action_space).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

    state, done = env.reset(), False
    state = resize(state).float().to(device)
    pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)

    steps = 0
    for step in pbar:
        with torch.no_grad():
            if step < UPDATE_START:
              # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
              action = torch.tensor(env.sample_action(), dtype=torch.float32, device=device).unsqueeze(dim=0)
            else:
              # Observe state s and select action a ~ μ(a|s)
              # action = actor(state).sample()
              action = actor(state).sample().float().to(device)

            # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
            next_state, reward, done = env.step(action.squeeze(dim=0).cpu())#.long
            next_state = resize(next_state).float().to(device)
            # Store (s, a, r, s', d) in replay buffer D
            # Store (s, a, r, s', d) in replay buffer D
            D.append({'state': state.unsqueeze(dim=0), 'action': action, 'reward': torch.tensor([reward],dtype=torch.float32,device=device), 'next_state': next_state.unsqueeze(dim=0), 'done': torch.tensor([True if reward == 1 else False], dtype=torch.float32, device=device)})
            state = next_state
            # If s' is terminal, reset environment state
            steps += 1

            if done or steps>step_limit: #TODO: incorporate step limit in the environment
                eval_c2 = True #TODO: multiprocess pyrep with a session for each testing and training
                steps = 0
                state = resize(env.reset()).float().to(device)

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
            batch = random.sample(D, BATCH_SIZE)
            state_batch = []
            action_batch = []
            reward_batch =  []
            state_next_batch = []
            done_batch = []
            for d in batch:
                state_batch.append(d['state'])
                action_batch.append(d['action'])
                reward_batch.append(d['reward'])
                state_next_batch.append(d['next_state'])
                done_batch.append(d['done'])

            batch = {'state':torch.cat(state_batch,dim=0),
                     'action':torch.cat(action_batch,dim=0),
                     'reward':torch.cat(reward_batch,dim=0),
                     'next_state':torch.cat(state_next_batch,dim=0),
                     'done':torch.cat(done_batch,dim=0)
                     }

            policy = actor(batch['state'])
            action, log_pi = policy.rsample_log_prob()
            #Automatic entropy tuning
            alpha_loss = -(log_alpha.float() * (log_pi + target_entropy).float().detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp()
            weighted_sample_entropy = (alpha.float() * log_pi).view(-1,1)

            # Compute targets for Q and V functions
            if VALUE_FNC:
                y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_value_critic(batch['next_state'])
                y_v = torch.min(critic_1(batch['state'], action.detach()), critic_2(batch['state'], action.detach())) - weighted_sample_entropy.detach()
            else:
                # No value function network
                next_policy = actor(batch['next_state'])
                next_actions, next_log_prob = next_policy.rsample_log_prob()
                target_qs = torch.min(target_critic_1(batch['next_state'], next_actions), target_critic_2(batch['next_state'], next_actions)) - alpha * next_log_prob
                y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_qs.detach()

            q_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()

            critics_optimiser.zero_grad()
            q_loss.backward()
            critics_optimiser.step()

            # Update V-function by one step of gradient descent
            if VALUE_FNC:
                v_loss = (value_critic(batch['state']) - y_v).pow(2).mean().to(device)

                value_critic_optimiser.zero_grad()
                v_loss.backward()
                value_critic_optimiser.step()

            # Update policy by one step of gradient ascent
            new_qs = torch.min(critic_1(batch["state"], action),critic_2(batch["state"], action))
            policy_loss = (weighted_sample_entropy - new_qs).mean().to(device)

            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()

            # Update target value network
            if VALUE_FNC:
                update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)
            else:
                update_target_network(critic_1, target_critic_1, POLYAK_FACTOR)
                update_target_network(critic_2, target_critic_2, POLYAK_FACTOR)

        if step > UPDATE_START and step % TEST_INTERVAL == 0: eval_c = True
        else: eval_c = False

        if eval_c == True and eval_c2 == True:
            eval_c = False
            eval_c2 = False
            actor.eval()
            critic_1.eval()
            critic_2.eval()
            q_value_eval = eval_.get_qvalue(critic_1,critic_2)
            return_ep, steps_ep = eval_.sample_episode(actor)

            logz.log_tabular('Training steps', step)
            logz.log_tabular('Validation return', return_ep.mean())
            logz.log_tabular('Validation steps',steps_ep.mean())
            logz.log_tabular('Validation return std',return_ep.std())
            logz.log_tabular('Validation steps std',steps_ep.std())
            logz.log_tabular('Q-value evaluation',q_value_eval)
            logz.log_tabular('Q-network loss', q_loss.detach().cpu().numpy())
            if VALUE_FNC:
                logz.log_tabular('Value-network loss', v_loss.detach().cpu().numpy())
            logz.log_tabular('Policy-network loss', policy_loss.detach().cpu().numpy())
            logz.log_tabular('Alpha loss',alpha_loss.detach().cpu().numpy())
            logz.log_tabular('Log Pi',log_pi.mean().detach().cpu().numpy())
            logz.dump_tabular()

            logz.save_pytorch_model(actor.state_dict())

            pbar.set_description('Step: %i | Reward: %f' % (step, return_ep.mean()))

            actor.train()
            critic_1.train()
            critic_2.train()

    env.terminate()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--ENV',required=True,type=str)
    parser.add_argument('--OBSERVATION_LOW',action='store_true')
    parser.add_argument('-bs','--BATCH_SIZE',default=64,type=int)
    parser.add_argument('--DISCOUNT',default=0.99,type=float)
    parser.add_argument('--ENTROPY_WEIGHT',default=0.2,type=float)
    parser.add_argument('--HIDDEN_SIZE',default=64,type=int)
    parser.add_argument('-lr','--LEARNING_RATE',default=0.002,type=float)
    parser.add_argument('-steps','--MAX_STEPS',default=100000,type=int)
    parser.add_argument('--POLYAK_FACTOR',default=0.995,type=float)
    parser.add_argument('--REPLAY_SIZE',default=100000,type=int)
    parser.add_argument('--TEST_INTERVAL',default=1000,type=int)
    parser.add_argument('--UPDATE_INTERVAL',default=1,type=int)
    parser.add_argument('--UPDATE_START',default=10000,type=int)
    parser.add_argument('--VALUE_FNC',action='store_true')

    args = parser.parse_args()
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    start_time = time.time()

    train(args.BATCH_SIZE,
          args.DISCOUNT,
          args.ENTROPY_WEIGHT,
          args.HIDDEN_SIZE,
          args.LEARNING_RATE,
          args.MAX_STEPS,
          args.POLYAK_FACTOR,
          args.REPLAY_SIZE,
          args.TEST_INTERVAL,
          args.UPDATE_INTERVAL,
          args.UPDATE_START,
          args.ENV,
          args.OBSERVATION_LOW,
          args.VALUE_FNC,
          logdir)

    print("Elapsed time: ", time.time() - start_time)

if __name__ == "__main__":
    main()
