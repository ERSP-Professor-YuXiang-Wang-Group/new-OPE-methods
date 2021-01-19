import tensorflow as tf
from abc import abstractmethod

class Basic_Alg():
    def __init__(self, scope):
        self.scope = scope

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def build_estimation_graph(self):
        pass

    @abstractmethod
    def create_loss_func(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass

    def get_percentile(self, x):
        ret = []
        for i in range(10):
            ret.append(tf.contrib.distributions.percentile(x, i * 10 + 5))
        return ret

    def get_all_vars(self):
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return tf.concat([tf.reshape(var, [-1]) for var in all_vars], axis=-1)