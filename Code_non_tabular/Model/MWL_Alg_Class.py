import tensorflow as tf
import numpy as np
import time
from Model.Basic_Alg_Class import Basic_Alg
 
class MWL(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, 
                       norm, q_net, med_dist,
                       hidden_layers=[32, 32], scope='mwl',
                       lr=5e-3, reg_factor=0, gamma=0.999):
        super().__init__(scope)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma #discounted factor
        self.lr = lr #learning rate
        self.reg_factor = reg_factor #regularization factor
        self.hidden_layers = hidden_layers

        self.med_dist = med_dist
        self.q_net = q_net
        self.norm = norm #norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}

        #XXX debug
        self.debug_w = {}

        self.trainable_vars = []

        self.build_graph()
        self.build_estimation_graph()
        self.create_loss_func()

        tf.get_default_session().run(
            [tf.variables_initializer(self.trainable_vars)]
        )
    
    def build_graph(self):
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.done_ph = tf.placeholder(dtype=tf.bool, shape=[None, 1])
        ''' Initial Part '''
        # ''' firs sample '''
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.init_act_b = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.init_act_e = self.build_action(self.init_obs_ph)

        self.init_obs_act_b = tf.concat([self.init_obs_ph, tf.cast(self.init_act_b, tf.float32)], axis=1)     
        self.init_obs_act_e = tf.concat([self.init_obs_ph, tf.cast(self.init_act_e, tf.float32)], axis=1)           
        
        # ''' second sample '''
        self.init_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.init_act_b_2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.init_act_e_2 = self.build_action(self.init_obs_ph_2)

        self.init_obs_act_b_2 = tf.concat([self.init_obs_ph_2, tf.cast(self.init_act_b_2, tf.float32)], axis=1)     
        self.init_obs_act_e_2 = tf.concat([self.init_obs_ph_2, tf.cast(self.init_act_e_2, tf.float32)], axis=1)           
        
        ''' Current Part '''
        # first sample
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.obs_act = tf.concat([self.obs_ph, tf.cast(self.act_ph, tf.float32)], axis=1)
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        
        # second sample
        self.obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph_2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.obs_act_2 = tf.concat([self.obs_ph_2, tf.cast(self.act_ph_2, tf.float32)], axis=1)
        self.factor_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        ''' Next Part '''
        # first sample
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])  
        self.next_act_b = tf.placeholder(dtype=tf.int32, shape=[None, 1])    
        self.next_obs_act_b = tf.concat([self.next_obs_ph, \
                tf.cast(self.next_act_b, tf.float32)], axis=1)
        self.next_act_e = [
            tf.zeros([tf.shape(self.next_obs_ph)[0], 1], dtype=tf.int32), #?
            tf.ones([tf.shape(self.next_obs_ph)[0], 1], dtype=tf.int32), #?
        ]
        self.next_obs_act_e = [
            tf.concat([self.next_obs_ph, tf.cast(self.next_act_e[0], tf.float32)], axis=1),
            tf.concat([self.next_obs_ph, tf.cast(self.next_act_e[1], tf.float32)], axis=1),
        ]
        self.prob_next = self.build_prob(self.next_obs_ph)

        # second part
        self.next_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.next_act_b_2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])    
        self.next_obs_act_b_2 = tf.concat([self.next_obs_ph_2, \
                tf.cast(self.next_act_b_2, tf.float32)], axis=1)
        self.next_act_e_2 = [
            tf.zeros([tf.shape(self.next_obs_ph_2)[0], 1], dtype=tf.int32),
            tf.ones([tf.shape(self.next_obs_ph_2)[0], 1], dtype=tf.int32),
        ]
        self.next_obs_act_e_2 = [
            tf.concat([self.next_obs_ph_2, tf.cast(self.next_act_e_2[0], tf.float32)], axis=1),
            tf.concat([self.next_obs_ph_2, tf.cast(self.next_act_e_2[1], tf.float32)], axis=1),
        ]
        self.prob_next_2 = self.build_prob(self.next_obs_ph_2)


        ''' Density Ratio '''
        # first part   
        self.w = self.create_density_ratio(self.obs_ph, self.act_ph, factor=self.factor, reuse=False)
        self.w_next = self.create_density_ratio(self.next_obs_ph, self.next_act_b, factor=self.factor, reuse=True)
        
        # second part     
        self.w_2 = self.create_density_ratio(self.obs_ph_2, self.act_ph_2, factor=self.factor_2, reuse=True)
        self.w_next_2 = self.create_density_ratio(self.next_obs_ph_2, self.next_act_b_2, factor=self.factor_2, reuse=True)

        self.w_init = self.create_density_ratio(self.init_obs_ph, self.init_act_b, reuse=True, normalize=True)
        self.w_init_2 = self.create_density_ratio(self.init_obs_ph_2, self.init_act_b_2, reuse=True, normalize=True)

    def build_action(self, obs_ph):
        # recover the original obs, in order to get the correct action prob
        if self.norm['type'] is not None:
            org_obs = obs_ph * self.norm['scale'] + self.norm['shift'] #?
        else:
            org_obs = obs_ph

        act = self.q_net.build_random_policy(org_obs, reuse=True)
        return tf.stop_gradient(act) #Didn't find anything in tf API

    def build_prob(self, obs_ph):
        # recover the original obs, in order to get the correct action prob
        if self.norm['type'] is not None:
            org_obs = obs_ph * self.norm['scale'] + self.norm['shift']
        else:
            org_obs = obs_ph

        return self.q_net.build_prob(org_obs, reuse=True)

    def create_density_ratio(self, obs_tf, act_tf, *, factor=None, reuse=False, normalize=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
            for h in self.hidden_layers:
                x = tf.layers.dense(x, h, activation=tf.nn.relu) #??

            w = tf.layers.dense(x, 1, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

            if reuse == False:
                self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

            w = tf.log(1 + tf.exp(w))

            if factor is not None:
                w = w * factor

            if normalize:
                w = w / tf.reduce_mean(w)
                
            return w

    def build_estimation_graph(self):
        rew = self.rew_ph
        w = self.create_density_ratio(self.obs_ph, self.act_ph, factor=self.factor, reuse=True, normalize=True)

        assert self.gamma < 1.0
        self.value_estimation = tf.reduce_mean(w * rew) / (1 - self.gamma) #Computes the mean of elements across dimensions of a tensor.
        
    ''' 
        create loss function, drop those term which do not depend pn w(s,a)
    '''
    def create_loss_func(self):
        coeff = [self.gamma ** 2] * 4 + \
                [self.gamma ** 2] * 1 + \
                [(1-self.gamma) ** 2] * 1 + \
                [(1-self.gamma) ** 2] * 1 + \
                [-self.gamma ** 2] * 4 + \
                [-self.gamma * (1 - self.gamma)] * 4 + \
                [self.gamma * (1 - self.gamma)] * 4 + \
                [self.gamma * (1 - self.gamma)] * 2 + \
                [-self.gamma * (1 - self.gamma)] * 2 + \
                [-(1 - self.gamma)**2] * 2

        Kernel = [
            # Term 1
            (self.next_obs_act_e[0], self.next_obs_act_e_2[0]),
            (self.next_obs_act_e[0], self.next_obs_act_e_2[1]),
            (self.next_obs_act_e[1], self.next_obs_act_e_2[0]),
            (self.next_obs_act_e[1], self.next_obs_act_e_2[1]),
            # Term 2
            (self.next_obs_act_b, self.next_obs_act_b_2),
            # Term 3
            (self.init_obs_act_b, self.init_obs_act_b_2),
            # Term 4
            (self.init_obs_act_e, self.init_obs_act_e_2),
            # Term 5
            (self.next_obs_act_e[0], self.next_obs_act_b_2),
            (self.next_obs_act_e[1], self.next_obs_act_b_2),
            (self.next_obs_act_b, self.next_obs_act_e_2[0]),
            (self.next_obs_act_b, self.next_obs_act_e_2[1]),
            # Term 6
            (self.next_obs_act_e[0], self.init_obs_act_b_2),
            (self.next_obs_act_e[1], self.init_obs_act_b_2),
            (self.init_obs_act_b, self.next_obs_act_e_2[0]),
            (self.init_obs_act_b, self.next_obs_act_e_2[1]),
            # Term 7
            (self.next_obs_act_e[0], self.init_obs_act_e_2),
            (self.next_obs_act_e[1], self.init_obs_act_e_2),
            (self.init_obs_act_e, self.next_obs_act_e_2[0]),
            (self.init_obs_act_e, self.next_obs_act_e_2[1]),
            # Term 8
            (self.next_obs_act_b, self.init_obs_act_b_2),
            (self.init_obs_act_b, self.next_obs_act_b_2),
            # Term 9
            (self.next_obs_act_b, self.init_obs_act_e_2),
            (self.init_obs_act_e, self.next_obs_act_b_2),
            # Term 10
            (self.init_obs_act_b, self.init_obs_act_e_2),
            (self.init_obs_act_e, self.init_obs_act_b_2),
        ]

        prob_mask = [
                # Term 1
                self.prob_next[0] * tf.reshape(self.prob_next_2[0], [1, -1]),
                self.prob_next[0] * tf.reshape(self.prob_next_2[1], [1, -1]),
                self.prob_next[1] * tf.reshape(self.prob_next_2[0], [1, -1]),
                self.prob_next[1] * tf.reshape(self.prob_next_2[1], [1, -1]),
            ] + \
            [ 
                # Term 2, 3, 4
                None
            ] * 3 + \
            [ 
                # Term 5, 6, 7
                self.prob_next[0] * tf.ones([1, tf.shape(self.prob_next_2[0])[0]]),
                self.prob_next[1] * tf.ones([1, tf.shape(self.prob_next_2[0])[0]]),
                tf.ones(tf.shape(self.prob_next[0])) * tf.reshape(self.prob_next_2[0], [1, -1]),
                tf.ones(tf.shape(self.prob_next[0])) * tf.reshape(self.prob_next_2[1], [1, -1]),
            ] + \
            [ 
                # Term 5, 6, 7
                self.prob_next[0] * tf.ones([1, tf.shape(self.w_init_2)[0]]),
                self.prob_next[1] * tf.ones([1, tf.shape(self.w_init_2)[0]]),
                tf.ones(tf.shape(self.w_init)) * tf.reshape(self.prob_next_2[0], [1, -1]),
                tf.ones(tf.shape(self.w_init)) * tf.reshape(self.prob_next_2[1], [1, -1]),
            ] * 2 + \
            [
                # Term 8, 9, 10
                None
            ] * 6

        w_ones = tf.ones(tf.shape(self.w_init))
        w_2_ones = tf.ones(tf.shape(self.w_init_2))
        weights = [
            # Term 1
            (self.w, self.w_2),
            (self.w, self.w_2),
            (self.w, self.w_2),
            (self.w, self.w_2),
            # Term 2
            (self.w_next, self.w_next_2),
            # Term 3
            (self.w_init, self.w_init_2),
            # Term 4
            None,
            # Term 5
            (self.w, self.w_next_2),
            (self.w, self.w_next_2),
            (self.w_next, self.w_2),
            (self.w_next, self.w_2),
            # Term 6
            (self.w, self.w_init_2),
            (self.w, self.w_init_2),
            (self.w_init, self.w_2),
            (self.w_init, self.w_2),
            # Term 7
            (self.w, w_2_ones),
            (self.w, w_2_ones),
            (w_ones, self.w_2),
            (w_ones, self.w_2),
            # Term 8
            (self.w_next, self.w_init_2),
            (self.w_init, self.w_next_2),
            # Term 9
            (self.w_next, w_2_ones),
            (w_ones, self.w_next_2),
            # Term 10
            (self.w_init, w_2_ones),
            (w_ones, self.w_init_2),
        ]

        self.reg_loss = self.reg_factor * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.scope))
        self.debug_w.update({'reg loss': self.reg_loss})
        self.loss = self.reg_loss

        K = 0
        for index in range(len(Kernel)):
            if weights[index] is None:
                continue

            k1, k2 = Kernel[index]
            c = coeff[index]
            w1, w2 = weights[index]

            diff = tf.expand_dims(k1, 1) - tf.expand_dims(k2, 0)
            K = tf.exp(-tf.reduce_sum(tf.square(diff), axis=-1)/2.0/self.med_dist[index] ** 2)
            if prob_mask[index] is not None:
                K = K * prob_mask[index]
            sample_num = tf.cast(tf.shape(K)[0] * tf.shape(K)[1], tf.float32)
            loss = c * tf.matmul(tf.matmul(tf.transpose(w1), K), w2) / sample_num

            self.loss += tf.squeeze(loss)#Removes dimensions of size 1 from the shape of a tensor.

        self.opt = tf.train.AdamOptimizer(self.lr) #Optimizer that implements the Adam algorithm(for Stochastic Optimization).
        self.train_op = self.opt.minimize(self.loss)

        self.trainable_vars += self.opt.variables()


    def train(self, data):
        debug, loss, _ = tf.get_default_session().run(
            [self.debug_w, self.loss, self.train_op],
            feed_dict={
                self.obs_ph: data['obs_1'],  
                self.act_ph: data['acts_1'],  
                self.next_obs_ph: data['next_obs_1'],
                self.next_act_b: data['next_acts_1'],
                self.init_obs_ph: data['init_obs_1'],
                self.init_act_b: data['init_acts_1'],
                self.factor: data['factor_1'],
                self.obs_ph_2: data['obs_2'],
                self.act_ph_2: data['acts_2'],
                self.next_obs_ph_2: data['next_obs_2'],
                self.next_act_b_2: data['next_acts_2'],
                self.init_obs_ph_2: data['init_obs_2'],
                self.init_act_b_2: data['init_acts_2'],
                self.factor_2: data['factor_2'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return debug, loss

    def evaluation(self, obs, acts, factor, rew):
        value = tf.get_default_session().run(
            self.value_estimation,
            feed_dict={
                self.obs_ph: obs,
                self.act_ph: acts,
                self.rew_ph: rew,
                self.factor: factor,#?
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value