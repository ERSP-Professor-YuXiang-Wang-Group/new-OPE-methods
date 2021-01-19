import tensorflow as tf
import numpy as np
import os

class classifier():
    def __init__(self, obs_dim, act_dim, lr=1e-4, *,
                    scope='classifer',
                    hidden_layers=[64, 64]):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.scope = scope
        
        self.hidden_layers = hidden_layers                

        self.debug = {}

        self.build_network()

        tf.get_default_session().run(
            [tf.variables_initializer(self.trainable_vars)]
        )
    
    def build_network(self, reuse=False):
        with tf.variable_scope(self.scope, reuse):
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
            self.labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
            self.labels = self.create_label(self.labels_ph, label_smooth=True)            

            x = self.obs_ph
            index = 0
            for h in self.hidden_layers:
                x = tf.contrib.layers.fully_connected(x, h, activation_fn=tf.nn.relu)
                index += 1

            self.logits = tf.contrib.layers.fully_connected(x, self.act_dim, activation_fn=None)
            
            # probability
            self.prob = tf.nn.softmax(self.logits, axis=1)
            mask = tf.one_hot(tf.squeeze(self.labels_ph), depth=self.act_dim)
            self.prob_given_act = tf.reduce_sum(self.prob * mask, axis=1, keep_dims=True)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.labels,
                    logits=self.logits,
                    axis=-1,        
                )
            )

            self.opt = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.opt.minimize(self.loss)

            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope) + self.opt.variables()            

    def get_prob_with_act(self, obs, act):
        return tf.get_default_session().run(
            self.prob_given_act,
            feed_dict={
                self.obs_ph: obs,
                self.labels_ph: act,
            }
        )

    def create_label(self, labels, label_smooth=True, epsilon=1e-3):
        target = tf.one_hot(tf.squeeze(labels), depth=self.act_dim)

        if label_smooth:
            target = target * (1 - 2 * epsilon) + epsilon

        return target

    def fit(self, data):
        loss, debug, _ = tf.get_default_session().run(
            [self.loss, self.debug, self.train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.labels_ph: data['acts'],
            }
        )
        return debug, loss