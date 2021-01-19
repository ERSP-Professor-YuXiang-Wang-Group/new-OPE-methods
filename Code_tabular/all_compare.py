import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
import quadprog
import os
import sys
import argparse
import optparse
import subprocess
import numpy as np
#######from Density_Ratio_discrete import Density_Ratio_discrete, Density_Ratio_discounted
from Q_learning import Q_learning
from environment import random_walk_2d, taxi


args = sys.argv
ver = np.int(args[1])
nt = np.int(args[2])
alpha = np.float(args[3])
ts = np.int(args[4])
gm = np.float(args[5])



def linear_solver(n, M):
	M -= np.amin(M)	# Let zero sum game at least with nonnegative payoff
	c = np.ones((n))
	b = np.ones((n))
	res = linprog(-c, A_ub = M.T, b_ub = b)
	w = res.x
	return w/np.sum(w)



def quadratic_solver(n, M, bbb ,tvalid, regularizer):
    qp_G = np.matmul(M.T, M)
    qp_G += regularizer * np.eye(n)

    qp_a = np.matmul(M.T, bbb)###np.zeros(n, dtype = np.float64)

    qp_C = np.zeros((n,n+1), dtype = np.float64)
    for i in range(n):
        if i in tvalid:
            qp_C[i,0] = 1.0
        qp_C[i,i+1] = 1.0
    qp_b = np.zeros(n+1, dtype = np.float64)
    qp_b[0] = 1.0
    meq =1
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    w = res[0]
    return w


'''
def quadratic_solver_extend(n, M, b, regularizer):
	qp_G = np.matmul(M, M.T)
	qp_G += regularizer * np.eye(n)
	
	qp_a = np.matmul(b[None, :], M.T).reshape(-1)

	qp_C = np.zeros((n,n+1), dtype = np.float64)
	for i in range(n):
		qp_C[i,0] = 1.0
		qp_C[i,i+1] = 1.0
	qp_b = np.zeros(n+1, dtype = np.float64)
	qp_b[0] = 1.0
	meq = 1
	res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
	w = res[0]
	return w
'''

class Density_Ratio_discounted(object):
    def __init__(self, num_state, gamma):
        self.num_state = num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)
        self.auxi = np.zeros([num_state, 1], dtype = np.float64)
        self.gamma = gamma
        #####self.initial_b = np.zeros([num_state], dtype = np.float64)self.gamma = gamma

    def reset(self):
        num_state = self.num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

    def feed_data(self, cur, next, initial, policy_ratio, discounted_t):
        if cur == -1:
            self.Ghat[next, next] -= 1.0
        else:
            self.Ghat[next, cur] += self.gamma*policy_ratio 
            self.Ghat[next, next] -= 1.0 
            self.auxi += -(1.0-self.gamma)*np.load("emp_hist.npy")
            self.Nstate[cur] += 1.0 #####discounted_t

    def density_ratio_estimate(self, regularizer = 0.0001):
        Frequency = self.Nstate.reshape(-1)
        auxi = self.auxi.reshape(-1)
        print self.auxi.shape
        tvalid = np.where(Frequency >= 1e-5)[0]
        G = np.zeros_like(self.Ghat)
        ####auxi = auxi*0.0
        Frequency = Frequency/np.sum(Frequency)
        auxi[tvalid] = self.auxi[tvalid].reshape(-1)#####*Frequency[tvalid]
        print G.shape
        G[:,tvalid] = self.Ghat[:,tvalid]/Frequency[tvalid]		

        n = self.num_state
       
       
        x = quadratic_solver(n, G/100.0,auxi/100.0,tvalid, 1.0*regularizer)
     
        #####res = minimize(loss_weight_w_mini,1.0/self.num_state*np.ones([self.num_state, 1] ))
        #####x = res.x
        print np.sum(x)
        w = np.zeros(self.num_state)
        w[tvalid] = x[tvalid]/Frequency[tvalid]        
        ###w[w>3.0] = 0.0
        return x, w



def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        sasr = []
        for i_t in range(truncate_size):
            #env.render()
            p_action = policy[state, :]
            action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward = env.step(action)

            sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            #print env.state_decoding(state)
            #a = input()

            state = next_state
        SASR.append(sasr)
    return SASR, frequency, total_reward/(num_trajectory * truncate_size)

def train_density_ratio(SASR, policy0, policy1, den_discrete, gamma):
    for sasr in SASR:
        discounted_t = 1.0
        initial_state = sasr[0,0]
        for state, action, next_state, reward in sasr:
            discounted_t = gamma
            policy_ratio = policy1[state, action]/policy0[state, action]
            den_discrete.feed_data(state, next_state, initial_state, policy_ratio, discounted_t)
        ####den_discrete.feed_data(-1, initial_state, initial_state, 1, discounted_t)

    x, w = den_discrete.density_ratio_estimate()
    return x, w

def off_policy_evaluation_density_ratio(SASR, policy0, policy1, density_ratio, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    for sasr in SASR:
        discounted_t = gamma
        for state, action, next_state, reward in sasr:
            policy_ratio = policy1[state, action]/policy0[state, action]
            total_reward += density_ratio[state] * policy_ratio * reward #### * discounted_t
            self_normalizer += 1.0###density_ratio[state] * policy_ratio ######* discounted_t
            #######discounted_t = gamma
    return total_reward / self_normalizer


def double_evaluation_density_ratio(SASR, policy0, policy1, density_ratio, gamma, q_table, pi):
    total_reward = 0.0
    self_normalizer = 0.0
    for sasr in SASR:
        discounted_t = gamma
        for state, action, next_state, reward in sasr:
            policy_ratio = policy1[state, action]/policy0[state, action]
            total_reward += density_ratio[state] * policy_ratio * (reward+gamma*np.sum(pi[next_state]*q_table[next_state,:]) -q_table[state,action]) #### * discounted_t
            self_normalizer += 1.0###density_ratio[state] * policy_ratio ######* discounted_t
            #######discounted_t = gamma
    return total_reward / self_normalizer

def on_policy(SASR, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    for sasr in SASR:
        discounted_t = 1.0
        for state, action, next_state, reward in sasr:
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
    return total_reward / self_normalizer

def importance_sampling_estimator(SASR, policy0, policy1, gamma):
    mean_est_reward = 0.0
    for sasr in SASR:
        log_trajectory_ratio = 0.0
        total_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        avr_reward = total_reward / self_normalizer
        mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
    mean_est_reward /= len(SASR)
    return mean_est_reward

def importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
    mean_est_reward = 0.0
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            step_log_pr += np.log(policy1[state, action]) - np.log(policy0[state, action])
            est_reward += np.exp(step_log_pr)*reward*discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        est_reward /= self_normalizer
        mean_est_reward += est_reward
    mean_est_reward /= len(SASR)
    return mean_est_reward

def double_importance_sampling_estimator(SASR, policy0, policy1, gamma, q_table,pi):
    mean_est_reward = 0.0
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            step_log_pr_previous = np.copy(step_log_pr)
            step_log_pr += np.log(policy1[state, action]) - np.log(policy0[state, action])
            est_reward += np.exp(step_log_pr)*(reward- q_table[state,action])*discounted_t
            est_reward += np.exp(step_log_pr_previous)*(np.sum(pi[state]*q_table[state,:]))*discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        est_reward /= self_normalizer
        mean_est_reward += est_reward
    mean_est_reward /= len(SASR)
    return mean_est_reward


def weighted_importance_sampling_estimator(SASR, policy0, policy1, gamma):
    total_rho = 0.0
    est_reward = 0.0
    for sasr in SASR:
        total_reward = 0.0
        log_trajectory_ratio = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_trajectory_ratio += np.log(policy1[state, action]) - np.log(policy0[state, action])
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        avr_reward = total_reward / self_normalizer
        trajectory_ratio = np.exp(log_trajectory_ratio)
        total_rho += trajectory_ratio
        est_reward += trajectory_ratio * avr_reward

    avr_rho = total_rho / len(SASR)
    return est_reward / avr_rho/ len(SASR)

def weighted_importance_sampling_estimator_stepwise(SASR, policy0, policy1, gamma):
    Log_policy_ratio = []
    REW = []
    for sasr in SASR:
        log_policy_ratio = []
        rew = []
        discounted_t = 1.0
        self_normalizer = 0.0
        for state, action, next_state, reward in sasr:
            log_pr = np.log(policy1[state, action]) - np.log(policy0[state, action])
            if log_policy_ratio:
                log_policy_ratio.append(log_pr + log_policy_ratio[-1])
            else:
                log_policy_ratio.append(log_pr)
            rew.append(reward * discounted_t)
            self_normalizer += discounted_t
            discounted_t *= gamma
        Log_policy_ratio.append(log_policy_ratio)
        REW.append(rew)
    est_reward = 0.0
    rho = np.exp(Log_policy_ratio)
    #print 'rho shape = {}'.format(rho.shape)
    REW = np.array(REW)
    for i in range(REW.shape[0]):
        est_reward += np.sum(rho[i]/np.mean(rho, axis = 0) * REW[i])/self_normalizer
    return est_reward/REW.shape[0]


def Q_learning(env, num_trajectory, truncate_size, temperature = 2.0):
    agent = Q_learning(n_state, n_action, 0.01, 0.99)

    state = env.reset()
    for k in range(20):
        print 'Training for episode {}'.format(k)
        for i in range(50):
            for j in range(5000):
                action = agent.choose_action(state, temperature)
                next_state, reward = env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state
        pi = agent.get_pi(temperature)
        np.save('taxi-policy/pi{}.npy'.format(k), pi)
        SAS, f, avr_reward = roll_out(n_state, env, pi, num_trajectory, truncate_size)
        print 'Episode {} reward = {}'.format(k, avr_reward)
        heat_map(length, f, env, 'heatmap/pi{}.pdf'.format(k))

"""
def model_based(n_state, n_action, SASR, pi, gamma):
    T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
    R = np.zeros([n_state, n_action], dtype = np.float32)
    R_count = np.zeros([n_state, n_action], dtype = np.int32)
    for sasr in SASR:
        for state, action, next_state, reward in sasr:
            T[state, action, next_state] += 1
            R[state, action] += reward
            R_count[state, action] += 1
    d0 = np.zeros([n_state, 1], dtype = np.float32)

    for state in SASR[:,0,0].flat:
        d0[state, 0] += 1.0
    ###d0 = np.ones([n_state, 1], dtype = np.float32)
    d0 = np.load("emp_hist.npy")
    t = np.where(R_count > 0)
    t0 = np.where(R_count == 0)
    R[t] = R[t]/R_count[t]
    R[t0] = np.mean(R[t])
    T = T + 1e-9	# smoothing
    T = T/np.sum(T, axis = -1)[:,:,None]
    Tpi = np.zeros([n_state, n_state])
    for state in range(n_state):
        for next_state in range(n_state):
            for action in range(n_action):
                Tpi[state, next_state] += T[state, action, next_state] * pi[state, action]
    dt = d0/np.sum(d0)
    dpi = np.zeros([n_state, 1], dtype = np.float32)
    truncate_size = SASR.shape[1]
    discounted_t = 1.0
    self_normalizer = 0.0
    for i in range(truncate_size):
        dpi += dt * discounted_t
        if i < 200:
            dt = np.dot(Tpi.T,dt)
        self_normalizer += discounted_t
        discounted_t *= gamma
    dpi /= self_normalizer
    np.save("Tpi.npy",Tpi)
    np.save("R.npy",R)
    np.save("pi.npy",pi)
    Rpi = np.sum(R * pi, axis = -1)
    np.save("Rpi.npy",Rpi)
    np.save("dpi.npy",dpi)
    return np.sum(dpi.reshape(-1) * Rpi)
"""
def model_based(n_state, n_action, SASR, pi, gamma):
    Q_table = np.zeros([n_state, n_action], dtype = np.float32)
    T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
    R = np.zeros([n_state, n_action], dtype = np.float32)
    R_count = np.zeros([n_state, n_action], dtype = np.int32)
    for sasr in SASR:
        for state, action, next_state, reward in sasr:
            state = np.int(state)
            action = np.int(action)
            next_state = np.int(next_state)
            T[state, action, next_state] += 1
            R[state, action] += reward
            R_count[state, action] += 1
    d0 = np.ones([n_state, 1], dtype = np.float32)

    t = np.where(R_count > 0)
    t0 = np.where(R_count == 0)
    R[t] = R[t]/R_count[t]
    ###R[t0] = np.mean(R[t])
    T = T + 1e-9	# smoothing
    T = T/np.sum(T, axis = -1)[:,:,None]
    
    ddd = np.load("emp_hist.npy").reshape(-1)
    ####ddd = d0/np.sum(d0)
    for i in range(2000):
        Q_table_new = np.zeros([n_state, n_action], dtype = np.float32)
        V_table = np.sum(Q_table*pi,1)
        for state in range(n_state):
            for action in range(n_action):
                Q_table_new[state,action] = R[state,action]+gamma*np.sum(T[state, action, :]*V_table)
        Q_table = np.copy(Q_table_new)
        
    return np.sum(np.sum(Q_table*pi,1).reshape(-1)*ddd)*(1-gamma), Q_table

def run_experiment(n_state, n_action, SASR, pi0, pi1, gamma):

    den_discrete = Density_Ratio_discounted(n_state, gamma)
    x, w = train_density_ratio(SASR, pi0, pi1, den_discrete, gamma)
    x = x.reshape(-1)
    w = w.reshape(-1)

    est_DENR = off_policy_evaluation_density_ratio(SASR, pi0, pi1, w, gamma)
    est_naive_average = 0.0 ####on_policy(SASR, gamma)
    est_IST = importance_sampling_estimator(SASR, pi0, pi1, gamma)
    est_ISS = importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma)
    
    est_WIST = 0.0 ####weighted_importance_sampling_estimator(SASR, pi0, pi1, gamma)
    est_WISS = 0.0 ######weighted_importance_sampling_estimator_stepwise(SASR, pi0, pi1, gamma)

    est_model_based, Q_table = model_based(n_state, n_action, SASR, pi1, gamma)
    ddd = np.load("emp_hist.npy").reshape(-1)
    est_model_double = np.sum(np.sum(Q_table*pi1,1).reshape(-1)*ddd)*(1-gamma)+double_evaluation_density_ratio(SASR, pi0, pi1, w, gamma, Q_table,pi1)
    est_drl = double_importance_sampling_estimator(SASR, pi0, pi1, gamma, Q_table, pi1)
    #Q-table misspecified 
    Q_table_mis = Q_table + np.random.normal(1.0,1.0,n_state*n_action).reshape([n_state,n_action])
    est_model_based_mis_q = np.sum(np.sum(Q_table_mis*pi1,1).reshape(-1)*ddd)*(1-gamma)
    est_model_double_mis_q = np.sum(np.sum(Q_table_mis*pi1,1).reshape(-1)*ddd)*(1-gamma)+double_evaluation_density_ratio(SASR, pi0, pi1, w, gamma, Q_table_mis,pi1)
    est_drl_mis_q = double_importance_sampling_estimator(SASR, pi0, pi1, gamma, Q_table_mis, pi1)
    ##### ratio misspecified 
    w_mis = w + np.random.normal(1.0,1.0,n_state) 
    est_DENR_mis_w = off_policy_evaluation_density_ratio(SASR, pi0, pi1, w_mis, gamma)
    est_model_double_mis_w = np.sum(np.sum(Q_table*pi1,1).reshape(-1)*ddd)*(1-gamma)+double_evaluation_density_ratio(SASR, pi0, pi1, w_mis, gamma, Q_table,pi1)
    
    return est_DENR, est_naive_average, est_IST, est_ISS,est_drl, est_WIST, est_WISS, est_model_based,est_model_double,est_model_based_mis_q, est_model_double_mis_q,est_drl_mis_q, est_DENR_mis_w,est_model_double_mis_w  

####if __name__ == '__main__':

estimator_name = ['On Policy', 'Density Ratio', 'Naive Average', 'IST', 'ISS', "DRL",'WIST', 'WISS', 'Model Based','double',"Model Base misq","Double misq","mis_q","DR wmis","Double misq"]
length = 5
env = taxi(length)
n_state = env.n_state
n_action = env.n_action



"""
parser = argparse.ArgumentParser(description='taxi environment')
parser.add_argument('--nt', type = int, required = False, default = num_trajectory)
parser.add_argument('--ts', type = int, required = False, default = truncate_size)
parser.add_argument('--gm', type = float, required = False, default = gamma)
args = parser.parse_args()
"""
behavior_ID = 4
target_ID = 5

pi_target = np.load('taxi-policy/pi19.npy')
pi_behavior = np.load('taxi-policy/pi3.npy')

pi_behavior = alpha * pi_target + (1-alpha) * pi_behavior

res = np.zeros((15, 20), dtype = np.float32)
######### 10 means repetetions. 
for k in range(10):
    print k
    np.random.seed(30*ver+k)
    SASR0, _, _ = roll_out(n_state, env, pi_behavior, nt, ts)
    res[1:,k] = run_experiment(n_state, n_action, np.array(SASR0), pi_behavior, pi_target, gm)

    np.random.seed(30*ver+k)
    SASR, _, _ = roll_out(n_state, env, pi_target, nt, ts)
    res[0, k] = on_policy(np.array(SASR), gm)

    print('------seed = {}------'.format(k))
    for i in range(15):
        print('  ESTIMATOR: '+estimator_name[i]+ ', rewards = {}'.format(res[i,k]))
        print('----------------------')
        sys.stdout.flush()
    np.save('result/ver={}nt={}ts={}gm={}.npy'.format(ver,nt,ts,gm), res)


