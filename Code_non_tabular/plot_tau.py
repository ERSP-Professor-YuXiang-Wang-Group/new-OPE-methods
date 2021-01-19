import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

from Logger.logger import logger_class

def get_parser():
    parser = argparse.ArgumentParser(description='MQL')
    parser.add_argument('--est', type = str, nargs='+',
                                default=['MQL', 'MWL', 'DualDICE', 'MSWL'], help='training iteration for MQL')
    parser.add_argument('--tau', type = float, nargs='+', 
                                default=[0.25, 0.5, 1.5, 2.0], help='training iteration for MQL')
    args = parser.parse_args()

    return args

def plot_tau(args=None):
    # get behavior policy's value
    with open('./log/OnPolicy/1.0/log.pickle', 'rb') as f:
        behavior_value = pickle.load(f)['True_Rew']
    
    if args is None:
        estimators = ['MQL', 'MWL', 'DualDICE', 'MSWL']
        tau = [0.25, 0.5, 1.5, 2.0]
    else:
        estimators = args.est
        tau = args.tau
        
    log_dir = './log'

    colors = ['r', 'b', 'g', 'orange', 'purple'][:len(estimators)]
    markers = ['o', 's', '^', 'd'][:len(estimators)]
    assert len(estimators) <= len(colors)

    log_mse_dict = {}
    error_bar_dict = {}
    for est in estimators:
        log_mse_dict[est] = []
        error_bar_dict[est] = []

    os.chdir(log_dir)
    for t in tau:
        os.chdir('OnPolicy')
        with open(os.path.join(str(t), 'log.pickle'), 'rb') as f:
            on_policy = pickle.load(f)['True_Rew']
        os.chdir('..')
        for est in estimators:
            os.chdir(os.path.join(est, '200'))
            value = []
            count = 0
            datasets = os.listdir()
            for d in datasets:
                if not os.path.exists(os.path.join(d, str(t))):
                    continue
                os.chdir(os.path.join(d, str(t)))
                for log in os.listdir():
                    with open(log, 'rb') as f:
                        avg_list = pickle.load(f)['Recent_5_Avg']
                        value.append((avg_list[-1] - on_policy) ** 2)
                        count += 1
                os.chdir('../..')
            value = np.array(value)
            mean_mse = np.mean(value) / (on_policy - behavior_value) ** 2
            log_mse_dict[est].append(np.log(mean_mse))
            scale = np.sqrt(len(value))

            ste = 2 * np.std(value / (on_policy - behavior_value) ** 2, ddof=1) / scale
            
            error_bar_dict[est].append(np.array([[np.log(mean_mse / (mean_mse - ste))], [np.log((mean_mse + ste) / mean_mse)]]))

            os.chdir('../..')

    x_coord = range(len(tau))
    for est, color, marker in zip(estimators, colors, markers):
        plt.scatter(x_coord, log_mse_dict[est], c='', linewidths=3, edgecolors=color, marker=marker, label=est, s=200)
        plt.errorbar(x_coord, log_mse_dict[est], linestyle='', yerr=np.concatenate(error_bar_dict[est], axis=-1), fmt=color, elinewidth=3.0, capsize=15)
        for index in range(len(x_coord) - 1):
            plt.plot(x_coord[index:index+2], log_mse_dict[est][index:index+2], '--', color=color)


    plt.xlabel('Value of ' + r'$\tau$', fontsize=25)
    plt.ylabel('log MSE (relative)', fontsize=25)
    plt.legend(prop={'size': 20}, loc='lower right')
    plt.xticks(range(len(tau)), tau, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Changing ' + r'$\tau$' + ' around 1.0', fontsize=30)


if __name__ == '__main__':
    args = get_parser()
    
    plt.figure(figsize=(12, 8))

    plot_tau(args)

    plt.show()

    plt.tight_layout()
    plt.savefig('./Changing_Tau.png')
    print('save to ', os.path.join(os.getcwd(), 'Changing_Tau.png'))
