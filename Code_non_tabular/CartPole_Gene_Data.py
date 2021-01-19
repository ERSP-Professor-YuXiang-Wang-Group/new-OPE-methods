import numpy as np
import gym
import pickle
import time
import os
import argparse
import utils as U

from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from utils import get_percentile


def generate_dataset(env, q_net, n_ros=200):
    act_choices = [0, 1]
    accum_rew = 0.0

    obs_list = []
    act_list = []
    next_obs_list = []
    next_acts_list = []
    rew_list = []
    done_list = []
    init_obs = []
    init_acts = []

    for i in range(n_ros):
        obs = env.reset()
        init_obs.append(obs)
        
        done = False

        obs_list.append(obs)
        accum_rew = 0

        is_init_step = True
        while True:
            act = q_net.sample_action([obs])

            act = act[0]
            
            next_obs, rew, done, _ = env.step([act_choices[act]])
            accum_rew += rew

            if is_init_step:
                is_init_step = False
                init_acts.append(act)
            else:
                next_acts_list.append(act)
                assert len(next_acts_list) == len(act_list), '{} {}'.format(len(next_acts_list), len(act_list))

            act_list.append(act)
            rew_list.append(rew)
            next_obs_list.append(next_obs)
            done_list.append(done)

            if done:
                act = q_net.sample_action([next_obs])
                next_acts_list.append(act[0])
                break
            obs = next_obs
            obs_list.append(obs)

        print('Trajectory Number: ', i, 'Reward: ', accum_rew)        

    '''
    Return Shape:
    obs: [None, obs_dim],
    acts: [None, 1],
    next_obs: [None, obs_dim],
    next_acts: [None, 1],
    rews: [None, 1],
    done: [None, 1],
    init_obs: [None, obs_dim],
    init_acts: [None, 1],
    '''
    return {
        'obs': np.array(obs_list),
        'acts': np.array(act_list)[:, np.newaxis],
        'next_obs': np.array(next_obs_list),
        'next_acts': np.array(next_acts_list)[:, np.newaxis],
        'rews': np.array(rew_list)[:, np.newaxis],
        'done': np.array(done_list)[:, np.newaxis],
        'init_obs': np.array(init_obs),
        'init_acts': np.array(init_acts)[:, np.newaxis],
    }

def main():
    parser = argparse.ArgumentParser(description='MWL with PG')
    parser.add_argument('--tau', type = float, default = 1.0, help='temperature')
    parser.add_argument('--seed', type = int, nargs='+', default = [100], help='seed')
    parser.add_argument('--n-ros', type = int, default = 200, help='number of rollouts')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')
    
    args = parser.parse_args()

    env_name = "CartPole"

    ep_len = args.ep_len
    tau = args.tau

    st_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()
    
    q_net = Q_network(st_dim, act_dim, seed=100, default_tau=tau)
    q_net.load_model('./CartPole_Model/Model')

    if not os.path.exists('./Dataset/{}'.format(args.n_ros)):
        os.makedirs('./Dataset/{}'.format(args.n_ros))

    for seed in args.seed:
        env = CartPoleEnv(max_ep_len=ep_len, seed=seed)
        dataset = generate_dataset(env, q_net, n_ros=args.n_ros)
        with open('./Dataset/{}/CartPole-ep{}-tau{}-n{}-seed{}.pickle'.format(args.n_ros, ep_len, tau, args.n_ros, seed), 'wb') as f:
            pickle.dump(dataset, f) 

if __name__ == '__main__':
    main()
