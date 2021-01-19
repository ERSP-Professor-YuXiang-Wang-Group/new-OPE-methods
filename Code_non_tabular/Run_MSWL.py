import numpy as np
import gym
import pickle
import argparse
import time
import multiprocessing

import utils as U
from Model.MSWL_Alg_Class import MSWL
from Model.Q_Network_Class import Q_network
from Model.Classifier import classifier
from utils import eval_policy_cartpole_infinite, get_percentile
from Env.CartPole import CartPoleEnv
from Logger.logger import logger_class
from multiprocessing import Pool
from copy import deepcopy

'''
python Run_MSWL.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500
'''


def get_parser():
    parser = argparse.ArgumentParser(description='MSWL')

    parser.add_argument('--iter', type = int, default = 5000, help='training iteration for MSWL')
    parser.add_argument('--lr', type = float, default = 5e-3, help='learning rate')
    parser.add_argument('--reg', type = float, default = 0, help='regularization factor')
    parser.add_argument('--bs', type = int, default = 500, help='batch size')
    parser.add_argument('--gamma', type = float, default = 0.999, help='discounted factor')

    # hyper-parameter for policy distribution approximation
    parser.add_argument('--iter-pi', type = int, default = 10000, help='training iteration for MSWL')
    parser.add_argument('--bs-pi', type = int, default = 2000, help='batch size')
    parser.add_argument('--lr-pi', type = float, default = 5e-4, help='learning rate')
    
    parser.add_argument('--k-tau', type = float, default = 1.0, help='kernel bandwidth temperature')
    parser.add_argument('--norm', type = str, default = 'std_norm', help='normalization type')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed to generate dataset')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
    parser.add_argument('--tau', type = float, nargs='+', default = [1.5], help='policy temperature')
    parser.add_argument('--n-ros', type = int, nargs='+', default = [200], help='# trajectories in dataset')   

    args = parser.parse_args()

    return args

def sample_data(dataset, sample_num):
    data_size = dataset['obs'].shape[0]
    init_size = dataset['init_obs'].shape[0]
    
    index_1 = np.random.choice(data_size, sample_num)
    index_2 = np.random.choice(data_size, sample_num + 10)
    
    init_index_1 = np.random.choice(init_size, sample_num)
    init_index_2 = np.random.choice(init_size, sample_num + 10)

    ratio_1 = dataset['ratio'][index_1]
    ratio_2 = dataset['ratio'][index_2]

    return {
        'obs_1': dataset['obs'][index_1],
        'obs_2': dataset['obs'][index_2],
        'next_obs_1': dataset['next_obs'][index_1],
        'next_obs_2': dataset['next_obs'][index_2],
        'ratio_1': ratio_1,
        'ratio_2': ratio_2,
        'init_obs_1': dataset['init_obs'][init_index_1],
        'init_obs_2': dataset['init_obs'][init_index_2],
        'factor_1': dataset['factor'][index_1],
        'factor_2': dataset['factor'][index_2],
        'rew': dataset['rews'][index_1],
        'done': dataset['done'][index_1],
    }

def est_med_dist(dataset):
    data = sample_data(dataset, 5000)
    obs_1 = data['obs_1']
    obs_2 = data['obs_2']

    m0 = np.median(np.sqrt(np.sum(np.square(obs_1[None, :, :] - obs_2[:, None, :]), axis = -1)))
    
    return np.array([m0] * 4)

def main(args):
    env_name = "CartPole"
    ep_len = args.ep_len
    seed = args.seed
    norm_type = args.norm

    # set seed
    U.set_seed(args.dataset_seed + seed)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    '''load evaluation policy'''
    q_net = Q_network(obs_dim, act_dim, seed=args.seed + 10, default_tau=args.tau)
    U.initialize_all_vars()
    q_net.load_model('./CartPole_Model/Model')

    clf = classifier(obs_dim, act_dim, lr=args.lr_pi, hidden_layers=[64, 64])

    file_name = './Dataset/{}/CartPole-ep1000-tau1.0-n{}-seed{}.pickle'.format(args.n_ros, args.n_ros, args.dataset_seed)

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
        
        for iter in range(args.iter_pi):
            index = np.random.choice(dataset['obs'].shape[0], args.bs_pi)
            data = {'obs': dataset['obs'][index], 'acts': dataset['acts'][index]}
            debug, loss = clf.fit(data)

            if iter % 100 == 0:
                print('-------------------------------------')
                print('Behavior Policy Approximation')
                print('Iter ', iter + 1)
                print('Loss ', loss)
                for d in debug:
                    print(d, ' ', debug[d])                    
                print('-------------------------------------\n\n')
            
        dataset['act_prob'] = clf.get_prob_with_act(dataset['obs'], dataset['acts'])
        tar_act_prob = q_net.get_prob_with_act(dataset['obs'], dataset['acts'], tau=args.tau)

        dataset['ratio'] = tar_act_prob / (dataset['act_prob'])        
        # We drop (1-gamma) here because it will be normalized when we build density ratio
        dataset['factor'] = np.array([args.gamma ** (i % 1000) for i in range(dataset['obs'].shape[0])]).reshape([-1, 1])

        env = CartPoleEnv(max_ep_len=ep_len, seed=seed)        
        
        if norm_type == 'std_norm':
            obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            obs_std = np.std(dataset['obs'], axis=0, keepdims=True)

            dataset['obs'] = (dataset['obs'] - obs_mean) / obs_std
            dataset['next_obs'] = (dataset['next_obs'] - obs_mean) / obs_std
            dataset['init_obs'] = (dataset['init_obs'] - obs_mean) / obs_std

            norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}

        else:
            norm = {'type': None, 'shift': None, 'scale': None}

    med_dist = est_med_dist(dataset) / args.k_tau
    
    mswl = MSWL(obs_dim, act_dim, q_net=q_net,
            hidden_layers=[32, 32], lr=args.lr, 
            reg_factor=args.reg, gamma=args.gamma, med_dist=med_dist)
        
    # simple estimation
    value_true = eval_policy_cartpole_infinite(env, q_net, ep_num=10, greedy=False, gamma=args.gamma)

    log_name = 'log_seed{}.pickle'.format(args.seed)
    logger = logger_class(path='./log/MSWL/{}/Dataset{}/{}'.format(args.n_ros, args.dataset_seed, args.tau), name=log_name, tau=args.tau, env_name=env_name, value_true=value_true)

    value_est_list = []
    for iter in range(args.iter):
        data = sample_data(dataset, args.bs)
        debug, loss = mswl.train(data)

        if iter % 100 == 0:
            print('-------------------------------------')
            print('Iter ', iter + 1)
            print('Loss ', loss)
            for d in debug:
                print(d, ' ', debug[d])
            value_est = mswl.evaluation(dataset['obs'], dataset['acts'], dataset['factor'], dataset['rews'])

            print('True: {}. Estimate: {}'.format(value_true, value_est))
            value_est_list.append(value_est)

            print('Avgerage Over Recent 5 Estimation is ', np.mean(value_est_list[-5:]))
            print('Avgerage Over Recent 10 Estimation is ', np.mean(value_est_list[-10:]))

            print('-------------------------------------\n\n')            
            logger.add(iter, value_est, np.mean(value_est_list[-5:]))
            logger.dump()


if __name__ == '__main__':
    args = get_parser()

    args_list = []
    for dataset_seed in args.dataset_seed:
        for tau in args.tau:
            for seed in args.seed:
                for n_ros in args.n_ros:
                    args_copy = deepcopy(args)
                    args_copy.dataset_seed = dataset_seed
                    args_copy.seed = seed
                    args_copy.tau = tau
                    args_copy.n_ros = n_ros
                    args_list.append(args_copy)

    with Pool(processes=len(args_list), maxtasksperchild=1) as p:
            p.map(main, args_list, chunksize=1)

