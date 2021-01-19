import numpy as np
import os
import pickle
import time

class logger_class():
    def __init__(self, path='.', name=None, tau=None, env_name=None, value_true=None):
        assert name is not None, 'log_file should have a name'
        self.dir_path = path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.path = os.path.join(self.dir_path, name)
        
        self.iter = []
        self.true_rew = []
        self.est_rew = []
        self.recent_5_avg = []

        assert tau is not None
        assert env_name is not None

        self.doc = {
            'Iter': self.iter,
            'tau': tau,
            'True_Rew': value_true,
            'env_name': env_name,
            'Est_Rew': self.est_rew,
            'Recent_5_Avg': self.recent_5_avg,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def add(self, iter, est_rew, avg=0.0):
        self.iter.append(iter)
        self.est_rew.append(est_rew)
        self.recent_5_avg.append(avg)

    def info(self, string):
        print(string)