import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
import quadprog

def linear_solver(n, M):
	M -= np.amin(M)	# Let zero sum game at least with nonnegative payoff
	c = np.ones((n))
	b = np.ones((n))
	res = linprog(-c, A_ub = M.T, b_ub = b)
	w = res.x
	return w/np.sum(w)

def quadratic_solver(n, M, bbb , regularizer):
	qp_G = np.matmul(M, M.T)
	qp_G += regularizer * np.eye(n)
	
	qp_a = np.matmul(M, bbb)###np.zeros(n, dtype = np.float64)

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

class Density_Ratio_discrete(object):
	def __init__(self, num_state):
		self.num_state = num_state
		self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
		self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

	def reset(self):
		num_state = self.num_state
		self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
		self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

	def feed_data(self, cur, next, policy_ratio):
		self.Ghat[cur, next] += policy_ratio
		self.Ghat[next, next] -= 1.0
		self.Nstate[cur] += 0.5
		self.Nstate[next] += 0.5

	def density_ratio_estimate_old(self):
		Frequency = (self.Nstate + 1e-5)
		Frequency = Frequency / np.sum(Frequency)
		G = self.Ghat / Frequency
		n = self.num_state
		x = quadratic_solver(n, G/100.0)
		#x2 = linear_solver(n, G)
		#print x
		#print x2
		#print np.sum((x-x2)*(x-x2))
		w = x/Frequency.reshape(-1)
		return x, w

	def density_ratio_estimate(self, regularizer = 0.001):
		Frequency = self.Nstate.flat
		tvalid = np.where(Frequency >= 1e-9)	
		G = np.zeros_like(self.Ghat)
		Frequency = Frequency/np.sum(Frequency)
		G[tvalid] = self.Ghat[tvalid]/(Frequency[:,None])[tvalid]		
		n = self.num_state
		x = quadratic_solver(n, G/50.0, regularizer)
		w = np.zeros(self.num_state)
		w[tvalid] = x[tvalid]/Frequency[tvalid]
		return x, w

	def density_ratio_estimate_exact(self):
		Frequency = self.Nstate.flat
		tvalid = np.where(Frequency >= 1e-9)	
		G = np.zeros_like(self.Ghat)
		Frequency = Frequency/np.sum(Frequency)
		G = self.Ghat[tvalid, tvalid]/(Frequency[:,None])[tvalid]
		G = G/np.linalg.norm(G, 'fro')
		n = Frequency[tvalid].shape[0]
		x = np.zeros(self.num_state)
		x[tvalid] = quadratic_solver(n, G)
		w = np.zeros(self.num_state)
		w[tvalid] = x[tvalid]/Frequency[tvalid]
		return x, w

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
			self.Ghat[cur, next] += self.gamma*policy_ratio 
			###self.Ghat[cur, initial] += (1-self.gamma)/self.gamma * discounted_t
			self.Ghat[next, next] -= 1.0 
                    	self.auxi += (1.0-self.gamma)*1.0/self.num_state*np.ones([self.num_state, 1], dtype = np.float64)
            ####self.Ghat[next, next] -= discounted_t
			self.Nstate[cur] += 1.0 #####discounted_t

	def density_ratio_estimate(self, regularizer = 0.001):
		Frequency = self.Nstate.reshape(-1)
       		auxi = self.auxi.reshape(-1)
		print self.auxi.shape
		tvalid = np.where(Frequency >= 1e-5)
		G = np.zeros_like(self.Ghat)
        	auxi = auxi*0.0
		Frequency = Frequency/np.sum(Frequency)
        	auxi[tvalid] = self.auxi[tvalid].reshape(-1)/Frequency[tvalid]
		G[tvalid] = self.Ghat[tvalid]/(Frequency[:,None])[tvalid]		

		n = self.num_state
		np.save("GGG.npy",G)
		np.save("aixo.npy",auxi)
		x = quadratic_solver(n, G/10000.0,-auxi/10000.0, 10*regularizer)
		w = np.zeros(self.num_state)
		w[tvalid] = x[tvalid]/Frequency[tvalid]
		####w[w>3.0] = 0.0
		return x, w

