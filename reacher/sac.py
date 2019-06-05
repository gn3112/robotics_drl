from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from env import Env
from models import Critic, SoftActor, create_target_network, update_target_network
from images_to_video import im_to_vid
import logz
import inspect
import time
import os

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# SAC implementation from spinning-up-basic

def setup_logger(logdir,locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(params)
def test(actor,step,env,continuous,vid):
    img_ep = []
    step_ep = 0
    with torch.no_grad():
        state, done, total_reward = env.reset(), False, 0
    while not done:
        if continuous:
            action = actor(state.to(device)).mean
        else:
            action_dstr = actor(state.to(device))  # Use purely exploitative policy at test time
            _, action = torch.max(action_dstr,0)

        step_ep += 1
        if step_ep > 60:
            break
        state, reward, done = env.step(action.squeeze(dim=0).long())
        total_reward += reward
        img_ep.append(env.render())
    vid.from_list(img_ep,step)
    return total_reward
def optimise(args):
    return None

def train(BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS,POLYAK_FACTOR, REPLAY_SIZE, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START,logdir):
    setup_logger(logdir, locals())
    continuous = True
    env = Env(continuous=continuous)
    vid = im_to_vid(logdir)
    actor = SoftActor(HIDDEN_SIZE, continuous=continuous).to(device)
    critic_1 = Critic(HIDDEN_SIZE, 1, state_action= True if continuous else False).to(device)
    critic_2 = Critic(HIDDEN_SIZE, 1, state_action= True if continuous else False).to(device)
    value_critic = Critic(HIDDEN_SIZE, 1).to(device)
    target_value_critic = create_target_network(value_critic).to(device)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
    value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
    D = deque(maxlen=REPLAY_SIZE)

    #Automatic entropy tuning init
    target_entropy = -np.prod(2).item() #??
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optimizer = optim.Adam(log_alpha, lr=LEARNING_RATE)


    state, done = env.reset(), False
    pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
    for step in pbar:
      with torch.no_grad():
        if step < UPDATE_START:
          # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
          if continuous:
              action = torch.tensor([(3 * random.random() - 1.5),(3 * random.random() - 1.5)]).unsqueeze(dim=0)
          else:
              action = torch.tensor(random.randrange(8)).unsqueeze(dim=0)
        else:
          # Observe state s and select action a ~ μ(a|s)
          #action = actor(state).sample()
          if continuous:
              action = actor(state.to(device)).sample()
          else:
              action_dstr = actor(state.to(device))
              _, action = torch.max(action_dstr,0)
              action = action.unsqueeze(dim=0).long()
        # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
        next_state, reward, done = env.step(action.squeeze(dim=0))#.long
        # Store (s, a, r, s', d) in replay buffer D
        D.append({'state': state.unsqueeze(dim=0).to(device), 'action': action.to(device), 'reward': torch.tensor([reward]).float().to(device), 'next_state': next_state.unsqueeze(dim=0).to(device), 'done': torch.tensor([done], dtype=torch.float32).to(device)})
        state = next_state
        # If s' is terminal, reset environment state
        if done:
            state = env.reset()

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

        #batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        # Compute targets for Q and V functions

        y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_value_critic(batch['next_state'])
        if continuous:
            policy = actor(batch['state'])
            action = policy.rsample()  # a(s) is a sample from μ(·|s) which is differentiable wrt θ via the reparameterisation trick

            #Automatic entropy tuning
            log_pi = policy.log_prob(action).sum(dim=1)
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            aplha_loss.backward()
            self.alpha_optimizer.step()
            alpha = log_alpha.exp()

            weighted_sample_entropy = (alpha * log_pi).view(-1,1)  # Note: in practice it is more numerically stable to calculate the log probability when sampling an action to avoid inverting tanh

            y_v = torch.min(critic_1(batch['state'], action.detach()), critic_2(batch['state'], action.detach())) - weighted_sample_entropy.detach()
        else:
            action_dstr = actor(batch['state'])
            weighted_sample_entropy = ENTROPY_WEIGHT * torch.log(action_dstr)  # Note: in practice it is more numerically stable to calculate the log probability when sampling an action to avoid inverting tanh
            a_idx = (batch['action']).view(-1,1).to(device)
            y_v = torch.min(critic_1(batch['state']).gather(1,a_idx), critic_2(batch['state']).gather(1,a_idx)) - weighted_sample_entropy.detach().gather(1,a_idx)

        # Update Q-functions by one step of gradient descent
        if continuous:
            value_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean().to(device)
        else:
            value_loss = ((critic_1(batch['state']).gather(1,a_idx) - y_q).pow(2).mean() + (critic_2(batch['state']).gather(1,a_idx) - y_q).pow(2).mean()).to(device)
        critics_optimiser.zero_grad()
        value_loss.backward()
        critics_optimiser.step()

        # Update V-function by one step of gradient descent
        if continuous:
            value_loss = (value_critic(batch['state']) - y_v).pow(2).mean().to(device)
        else:
            value_loss = ((value_critic(batch['state']) - y_v).pow(2).mean()).to(device)
        value_critic_optimiser.zero_grad()
        value_loss.backward()
        value_critic_optimiser.step()

        # Update policy by one step of gradient ascent
        if continuous:
            policy_loss = (weighted_sample_entropy - critic_1(batch['state'], action)).mean().to(device)
        else:
            policy_loss = ((weighted_sample_entropy - critic_1(batch['state'])).sum(dim=1).mean()).to(device)
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()

        # Update target value network
        update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)

      if step > UPDATE_START and step % TEST_INTERVAL == 0:
        actor.eval()
        total_reward = test(actor, step, env, continuous,vid)
        logz.log_tabular('Step', step )
        logz.log_tabular('Validation total_reward', total_reward)
        logz.dump_tabular()
        pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))

        actor.train()

    env.terminate()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
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

    args = parser.parse_args()
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

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
          logdir)

if __name__ == "__main__":
    main()
