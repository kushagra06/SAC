import numpy as np
import gym
import gym.spaces
import torch
import torch.optim as optim
import torch.nn as nn

from gym_utils import get_state_dimension, get_action_dimension
from replay_buffer import ReplayBuffer
from model import VCritic, SoftQCritic, SACActor, NormalizedActions
from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt

mean_lambda = 1e-3
std_lambda = 1e-3
z_lambda = 0.0

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

def get_loss(val, next_val):
    criterion = nn.MSELoss()

    return criterion(val, next_val)

def send_to_device(s, a, r, next_s, done):
    s = torch.FloatTensor(s).to(device)
    a = torch.FloatTensor(a).to(device)
    r = torch.FloatTensor(r).unsqueeze(1).to(device)
    next_s = torch.FloatTensor(next_s).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    return s, a, r, next_s, done

def run_episode(actor, buffer, v_critic, target_v_critic, q_critic, env, gamma, freq, max_steps, global_step, v_optimizer, q_optimizer, actor_optimizer, evaluation=False):
    state = env.reset()
    done = False
    total_reward = 0
    #train_rewards = []
    #test_rewards = []
    for _ in range(max_steps):
        action = actor.get_action(state)
        next_state, reward, done, _ = env.step(action)
        if not evaluation:
            buffer.add(state, action, reward, next_state, done)

            if 256 < len(buffer):
                s, a, r, next_s, d = buffer.sample_batch(256)
                s, a, r, next_s, d = send_to_device(s, a, r, next_s, d) 
                
                q = q_critic.forward(s, a)
                v = v_critic.forward(s)
                new_a, log_prob, z, mean, log_std = actor.evaluate(s)

                target_v = target_v_critic.forward(next_s)
                next_q = r + (1 - d) * gamma * target_v
                q_loss = get_loss(q, next_q.detach())

                new_q = q_critic.forward(s, new_a)
                next_v = new_q - log_prob
                v_loss = get_loss(v, next_v.detach())

                log_prob_target = new_q - v
                actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

                #regularization losses
                mean_loss = mean_lambda * mean.pow(2).mean()
                std_loss = std_lambda * log_std.pow(2).mean()
                z_loss = z_lambda * z.pow(2).sum(1).mean()
                actor_loss += mean_loss + std_loss + z_loss

                q_critic.train(q_loss, q_optimizer)
                v_critic.train(v_loss, v_optimizer)
                actor.train(actor_loss, actor_optimizer)
                    
                #soft updates
                for target_param, param in zip(target_v_critic.parameters(), v_critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 5*1e-3) + param.data * 5*1e-3)

            global_step += 1
            state = next_state
            total_reward += reward


           #if global_step % freq == 0 and not evaluation:
                #reward, _, _ = run_episode(actor, buffer, v_critic, target_v_critic, q_critic, env, gamma, freq, max_steps, global_step, v_optimizer, q_optimizer, actor_optimizer, True)
                #test_rewards.append(reward)

            if done:
                break
        
    #train_rewards.append(total_reward)

    return total_reward, global_step




def run_experiment(actor, buffer, v_critic, target_v_critic, q_critic, env, freq, v_optimizer, q_optimizer, actor_optimizer, gamma=0.99, max_steps=500, n_epi=10000):
    global_step = 0
    test_rewards = []
    #test_rewards = [run_episode(actor, buffer, v_critic, target_v_critic, q_critic, env, gamma, freq, max_steps, global_step, v_optimizer, q_optimizer, actor_optimizer, evaluation=True)[0]]
    train_rewards = []
    num_steps = [0]
    for i in trange(n_epi):
        total_reward, global_step = run_episode(actor, buffer, v_critic, target_v_critic, q_critic, env, gamma, freq, max_steps, global_step, v_optimizer, q_optimizer, actor_optimizer, evaluation=False)
        train_rewards.append(total_reward)
        num_steps.append(global_step)
   #     test_rewards.extend(test_reward)
   
    return train_rewards, num_steps


def main(arg):
    train_rewards = []
    num_steps = []
    test_rewards = np.zeros(shape=(arg.n_exp, arg.max // arg.freq + 1), dtype=np.float64)

    for i in trange(arg.n_exp):
        env = NormalizedActions(gym.make(arg.env))
        env.seed(np.random.randint(12345))
        a_dim = get_action_dimension(env)            
        s_dim = get_state_dimension(env)

        buffer = ReplayBuffer(size=arg.buffer, a_dim=a_dim, a_dtype=np.float32, s_dim=s_dim, s_dtype=np.float32, store_mu=False)

        v_critic = VCritic().to(device)
        target_v_critic = VCritic().to(device)

        softq_critic = SoftQCritic().to(device)

        SAC_actor = SACActor().to(device)

        v_optimizer = optim.Adam(v_critic.parameters(), lr=3e-4) 
        q_optimizer = optim.Adam(softq_critic.parameters(), lr=3e-4) 
        actor_optimizer = optim.Adam(SAC_actor.parameters(), lr=3e-4) 

        train_rewards, steps = run_experiment(SAC_actor, buffer, v_critic, target_v_critic, softq_critic, env, arg.freq, v_optimizer, q_optimizer, actor_optimizer, arg.discount, arg.max, arg.n_epi)
        train_rewards.append(reward)
        num_steps.append(steps)


if __name__ == '__main__':
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('-b', '--buffer', type=int, default=1000000, help='Buffer size (default: 1000000).')
    argparser.add_argument('-d', '--discount', type=float, default=0.99, help='Discount factor, gamma (default: 0.99).')
    argparser.add_argument('-e', '--env', type=str, default='Pendulum-v0', help='ID of gym environment (default: Pendulum-v0).')
    argparser.add_argument('-f', '--freq', type=int, default=500, help='An evaluation episode is done at every _freq_ step.')
    argparser.add_argument('-m', '--max', type=int, default=500, help='Max number of steps (default: 500).')
    argparser.add_argument('-n', '--n_exp', type=int, default=20, help='Number of experiments (default: 20).')
    argparser.add_argument('-p', '--n_epi', type=int, default=10000, help='Number of episodes (default: 10000).')
    arg = argparser.parse_args()

    main(arg)


