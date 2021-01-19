import tensorflow as tf
import numpy as np
import time
from Model.Basic_Alg_Class import Basic_Alg

class MSWL(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, 
                       q_net, med_dist, 
                       hidden_layers=[32, 32], scope='mswl',
                       lr=5e-3, reg_factor=0, gamma=0.999):
        super().__init__(scope)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.q_net = q_net
        self.reg_factor = reg_factor
        self.hidden_layers = hidden_layers
        
        self.med_dist = med_dist

        #used to debug
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
        
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.pr_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.w = self.create_density_ratio(self.obs_ph, factor=self.factor, reuse=False)
        self.w_next = self.create_density_ratio(self.next_obs_ph, factor=self.factor, reuse=True)
        self.w_init = self.create_density_ratio(self.init_obs_ph, reuse=True, normalize=True)

        self.init_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.pr_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])

        self.factor_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.w_2 = self.create_density_ratio(self.obs_ph_2, factor=self.factor_2, reuse=True)
        self.w_next_2 = self.create_density_ratio(self.next_obs_ph_2, factor=self.factor_2, reuse=True)
        self.w_init_2 = self.create_density_ratio(self.init_obs_ph_2, reuse=True, normalize=True)

    def create_density_ratio(self, obs_tf, *, factor=None, reuse=False, normalize=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            x = obs_tf
            for h in self.hidden_layers:
                x = tf.layers.dense(x, h, activation=tf.nn.relu)
            w = tf.layers.dense(x, 1, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.))

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
        w = self.create_density_ratio(self.obs_ph, factor=self.factor, reuse=True, normalize=False)
        T = w * self.pr_ph

        assert self.gamma < 1.0
        self.value_estimation = tf.reduce_sum(T * rew) / tf.reduce_sum(T) / (1 - self.gamma)        

    def create_loss_func(self):
        coeff = [
            self.gamma ** 2,
            (1 - self.gamma) ** 2,
            self.gamma * ( 1 - self.gamma),
            self.gamma * ( 1 - self.gamma),
        ]
        Kernel = [
            (self.next_obs_ph, self.next_obs_ph_2),
            (self.init_obs_ph, self.init_obs_ph_2),
            (self.next_obs_ph, self.init_obs_ph_2),
            (self.init_obs_ph, self.next_obs_ph_2)
        ]

        w1 = self.w * self.pr_ph - self.w_next
        w2 = self.w_2 * self.pr_ph_2 - self.w_next_2

        w_init_1 = 1 - self.w_init
        w_init_2 = 1 - self.w_init_2        
        weights = [
            (w1, w2),
            (w_init_1, w_init_2),
            (w1, w_init_2),
            (w_init_1, w2),
        ]

        self.reg_loss = self.reg_factor * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.scope))        
        self.loss = self.reg_loss

        for index in range(len(Kernel)):
            c = coeff[index]
            k1, k2 = Kernel[index]
            x1, x2 = weights[index]

            diff = tf.expand_dims(k1, 1) - tf.expand_dims(k2, 0)
            K = tf.exp(-tf.reduce_sum(tf.square(diff), axis=-1)/2.0 / self.med_dist[index] ** 2)

            sample_num = tf.cast(tf.shape(K)[0] * tf.shape(K)[1], tf.float32)
            self.loss += tf.squeeze(c * tf.matmul(tf.matmul(tf.transpose(x1), K), x2)) / sample_num

        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss)

        self.trainable_vars += self.opt.variables()


    def train(self, data):
        debug, loss, _ = tf.get_default_session().run(
            [self.debug_w, self.loss, self.train_op],
            feed_dict={
                self.obs_ph: data['obs_1'],    
                self.obs_ph_2: data['obs_2'],
                self.next_obs_ph: data['next_obs_1'],
                self.next_obs_ph_2: data['next_obs_2'],
                self.pr_ph: data['ratio_1'],     
                self.pr_ph_2: data['ratio_2'],       
                self.init_obs_ph: data['init_obs_1'],
                self.init_obs_ph_2: data['init_obs_2'],
                self.factor: data['factor_1'],
                self.factor_2: data['factor_2'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return debug, loss

    def evaluation(self, obs, pr, factor, rew):
        value = tf.get_default_session().run(
            self.value_estimation,
            feed_dict={
                self.obs_ph: obs,
                self.rew_ph: rew,
                self.pr_ph: pr,
                self.factor: factor,
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value
    