import numpy as np
import tensorflow as tf
 
def make_session(num_cpu=1):
    """Returns a session that will use <num_cpu> CPU's only"""
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        gpu_options=gpu_options)
    return tf.Session(config=tf_config)

def initialize_all_vars():
    tf.get_default_session().run(tf.global_variables_initializer())

def set_seed(seed):
    tf.set_random_seed(seed + 1)
    np.random.seed(seed + 2)

def eval_policy_cartpole_infinite(env, q_net, ep_num=10, greedy=False, gamma=None):
    act_choices = [0, 1]
    accum_rew = 0.0
    rew_list = []

    for i in range(ep_num):
        obs = env.reset()
        done = False
        factor = 1.0

        while not done:
            act = q_net.sample_action([obs])[0]        
            obs, rew, done, _ = env.step([act_choices[act]])
            
            rew *= factor
            factor *= gamma

            accum_rew += rew
            rew_list.append(rew)

        print('Traj ', i, ' Current Average ', accum_rew / (i + 1))
    return accum_rew / ep_num


def get_percentile(data):
    ptr = []
    for i in range(10):
        ptr.append(np.percentile(data, i * 10 + 5))
    print(ptr)
