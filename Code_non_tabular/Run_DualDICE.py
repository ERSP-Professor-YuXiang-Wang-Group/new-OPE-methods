"""Run off-policy policy evaluation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import pickle
import argparse

import numpy as np
import tensorflow as tf

import Model.Neural_DualDICE as neural_dual_dice
import utils as U
import multiprocessing
from multiprocessing import Pool

from Env.CartPole import CartPoleEnv
from Model.Q_Network_Class import Q_network
from Logger.logger import logger_class
from copy import deepcopy

'''
python Run_DualDICE.py --tau 0.25 0.5 1.5 2.0 --dataset-seed 100 200 300 400 500
'''

def get_parser():
  parser = argparse.ArgumentParser(description='DualDICE')

  parser.add_argument('--num-trajectories', type = int, default = 200, help='Number of trajectories to collect.')
  parser.add_argument('--max-trajectory-length', type = int, default = 1000, help='Cutoff trajectory at this step.')
  
  parser.add_argument('--gamma', type = float, default = 0.999, help='Discount factor.')
  parser.add_argument('--norm', type = str, default = 'std_norm', help='Normalization Type')
  
  parser.add_argument('--solver-name', type = str, default = 'dice', help='Type of solver to use.')
  parser.add_argument('--save-dir', type=str, default = None, help='Directory to save results to.')
   
  parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')

  parser.add_argument('--function-exponent', type=float, default=1.5, help='Exponent for f function in DualDICE.')
  parser.add_argument('--deterministic-env', type=bool, default=False, help='assume deterministic env.')
  parser.add_argument('--batch-size', type=int, default=500, help='batch_size for training models.')

  parser.add_argument('--num-steps', type=int, default=30000, help='num_steps for training models.')
  parser.add_argument('--log-every', type=int, default=100, help='log after certain number of steps.')

  parser.add_argument('--nu-learning-rate', type=float, default=0.0005, help='nu lr')
  parser.add_argument('--zeta-learning-rate', type=float, default=0.005, help='z lr')
  
  parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed to generate dataset')
  parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
  parser.add_argument('--tau', type = float, nargs='+', default = [1.5], help='policy temperature')
  parser.add_argument('--n-ros', type = int, nargs='+', default = [200], help='# trajectories in dataset')   

  args = parser.parse_args()

  return args


def count_state_frequency(data, num_states, gamma):
  state_counts = np.zeros([num_states])
  for transition_tuple in data.iterate_once():
    state_counts[transition_tuple.state] += gamma ** transition_tuple.time_step
  return state_counts / np.sum(state_counts)

def main(args):
  seed = args.seed
  num_trajectories = args.num_trajectories
  max_trajectory_length = args.max_trajectory_length
  tau = args.tau
  gamma = args.gamma
  nu_learning_rate = args.nu_learning_rate
  zeta_learning_rate = args.zeta_learning_rate
  norm_type = args.norm
  
  solver_name = args.solver_name
  save_dir = args.save_dir
  
  dataset_seed = args.dataset_seed
  n_ros = args.n_ros
  
  # fix randomness
  U.set_seed(dataset_seed + seed)

  hparam_format = ('{ENV}_{TAU}_{NUM_TRAJ}_{TRAJ_LEN}_'
                   '{N_LR}_{Z_LR}_{GAM}_{SOLVER}')
  solver_str = (solver_name + '-%.1f' % args.function_exponent)
  hparam_str = hparam_format.format(
      ENV='CartPole',
      TAU=tau,
      NUM_TRAJ=num_trajectories,
      TRAJ_LEN=max_trajectory_length,
      GAM=gamma,
      N_LR=nu_learning_rate,
      Z_LR=zeta_learning_rate,
      SOLVER=solver_str)

  if save_dir:
    summary_dir = os.path.join(save_dir, hparam_str)
    summary_dir = os.path.join(summary_dir, 'seed%d' % start_seed)
    summary_writer = tf.summary.FileWriter(summary_dir)
  else:
    summary_writer = None

  # args = get_parser()

  env = CartPoleEnv(max_ep_len=args.ep_len, seed=seed)

  sess = U.make_session()
  sess.__enter__()

  obs_dim = 4
  act_dim = 2

  q_net = Q_network(obs_dim, act_dim, seed=args.seed + 10, default_tau=tau)
  U.initialize_all_vars()
  q_net.load_model('./CartPole_Model/Model')

  results = []
  
  summary_prefix = ''

  print('Get Started')
  file_name = './Dataset/{}/CartPole-ep1000-tau1.0-n{}-seed{}.pickle'.format(args.n_ros, args.n_ros, args.dataset_seed)

  with open(file_name, 'rb') as f:
    dataset = pickle.load(f)
    data_size = dataset['obs'].shape[0]
    dataset['time_step'] = np.array([i % 1000 for i in range(data_size)])
    dataset['acts'] = np.squeeze(dataset['acts'])
    dataset['next_acts'] = np.squeeze(dataset['next_acts'])
    dataset['rews'] = np.squeeze(dataset['rews'])

    if norm_type == 'std_norm':
        obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
        obs_std = np.std(dataset['obs'], axis=0, keepdims=True)

        dataset['obs'] = (dataset['obs'] - obs_mean) / obs_std
        dataset['next_obs'] = (dataset['next_obs'] - obs_mean) / obs_std
        dataset['init_obs'] = (dataset['init_obs'] - obs_mean) / obs_std

        norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}

    else:
        norm = {'type': None, 'shift': None, 'scale': None}

  # Get solver.
  neural_solver_params = neural_dual_dice.NeuralSolverParameters(
      obs_dim,
      act_dim,
      gamma,
      hidden_dim=32,
      hidden_layers=2,
      discrete_actions=True,
      deterministic_env=False,
      nu_learning_rate=args.nu_learning_rate,
      zeta_learning_rate=args.zeta_learning_rate,
      batch_size=args.batch_size,
      num_steps=args.num_steps,
      log_every=args.log_every,
      summary_writer=summary_writer,
      summary_prefix=summary_prefix)

  density_estimator = neural_dual_dice.NeuralDualDice(
      parameters=neural_solver_params,
      solve_for_state_action_ratio=True,
      function_exponent=args.function_exponent)

  log_name = 'log_seed{}.pickle'.format(seed)
  logger = logger_class(path='./log/DualDICE/{}/Dataset{}/{}'.format(n_ros, dataset_seed, tau), name=log_name, tau=tau, env_name='CartPole')

  # Solve for estimated density ratios.
  est_avg_rewards = density_estimator.solve(dataset, q_net, norm, logger)
  # Close estimator properly.
  density_estimator.close()
  print('Estimated (solver: %s) average step reward' % solver_name,
        est_avg_rewards)
  results.append([est_avg_rewards])

  if save_dir is not None:
    filename = os.path.join(save_dir, '%s.npy' % hparam_str)
    print('Saving results to %s' % filename)
    if not tf.gfile.IsDirectory(save_dir):
      tf.gfile.MkDir(save_dir)
    with tf.gfile.GFile(filename, 'w') as f:
      np.save(f, np.array(results))
  print('Done!')

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