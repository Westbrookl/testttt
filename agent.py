
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf

import numpy as np
import random_graph_gen as rd
import network_topology_gen as nt
# import memory_pool as mp
# import gcn_layer_with_pooling_v2 as gcn
import environment_gen as en
import time
import random

ENTROPY_WEIGHT = 0.5#shold be a placeholder for decaying
EPS = 1e-6
INPUT_FEATURES = 5
EXTRACTED_FEATURES = INPUT_FEATURES *8
SNODE_SIZE = 100
VNODE_FEATURES_SIZE = 3
NUM_AGENT = 4
ORDERS = 3
GAMMA=0.99
ALIVE_TIME=50000
config=tf.ConfigProto(
  device_count={"CPU":8},
  inter_op_parallelism_threads=1,
  intra_op_parallelism_threads=1
)


G,node_attr,min,max= rd.make_weighted_random_graph(SNODE_SIZE, 0.1, "normal",1)
LAPLACIAN = rd.make_laplacian_matrix(G)
LAPLACIAN_LIST = rd.make_laplacian_list(G, SNODE_SIZE, ORDERS)
LAPLACIAN_TENSOR = np.stack(LAPLACIAN_LIST)

log_dir = "/home/yanzx1993/VNR_A3C_SUMMARY"
#histogram:distribution
#scalar:scalar


def weight_init(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)


def bias_init(shape):
    #bias = tf.constant(0.1, shape=shape)
    bias = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(bias)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def variable_sum(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



class Agent(object):
    def __init__(self, name, environment=None):
        self.name = name
        # self.environment=environment
        if environment is None:
            self.environment = en.Environment("haha", 200, 50000)
        else:
            self.environment=environment
        self.state = self.get_state()
        self.substrate_size = self.environment.substrate_network.node_size
        self.action_space = np.arange(0, self.substrate_size, 1)

    def get_state(self):
        return self.environment.get_state()

    def perform_action(self, action):
        return self.environment.perform_action(action)

    def make_decision(self):
        return 0


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def td_target(v,r):
    return GAMMA*v+r


class ActorNetwork(object):
    def __init__(self, sess, name, input_features, snode_size, extracted_features, vnode_features_size, orders,
                 learning_rate=0.0001, act=tf.nn.relu):
        self.sess = sess
        self.name = name
        self.act = act
        self.input_features = input_features
        self.snode_size = snode_size
        self.extracted_features = extracted_features
        self.vnode_features_size = vnode_features_size
        self.orders = orders

        self.td_target = tf.placeholder(tf.float32, [1])
        self.selected_action = tf.placeholder(tf.float32, [self.snode_size])

        self.learning_rate = learning_rate

        self.snode_features, self.vnode_features, self.laplacian, self.action_prob, self.out, self.max_out = self.create_actor_network()

        self.network_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor")

        self.input_network_parameters = []
        for param in self.network_parameters:
            self.input_network_parameters.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_parameters_op = []
        for idx, param in enumerate(self.input_network_parameters):
            self.set_network_parameters_op.append(self.network_parameters[idx].assign(param))

        self.objective = tf.reduce_sum(tf.multiply(tf.log(
            tf.reduce_sum(tf.multiply(self.action_prob, self.selected_action), reduction_indices=1, keepdims=True)),
            -self.td_target)) + ENTROPY_WEIGHT * tf.reduce_sum(
            tf.multiply(self.action_prob, tf.log(self.action_prob + EPS)))

        self.entropy= ENTROPY_WEIGHT * tf.reduce_sum(
            tf.multiply(self.action_prob, tf.log(self.action_prob + EPS)))
        # self.objective=tf.reduce_sum(tf.multiply(self.action_prob, self.selected_action))

        self.actor_gradients = tf.gradients(self.objective, self.network_parameters)

        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.network_parameters))

    def create_actor_network(self):
        with tf.variable_scope("actor"):
            with tf.name_scope(self.name):
                with tf.name_scope("snode_features"):
                    snode_features = tf.placeholder(tf.float32, [self.input_features, self.snode_size])
                vnode_features = tf.placeholder(tf.float32, [self.vnode_features_size])
                laplacian = tf.placeholder(tf.float32, [self.orders, self.snode_size, self.snode_size])

                kernel_weights = weight_init([self.input_features, self.extracted_features, self.orders])
                feature_fc_weights = weight_init([self.extracted_features * self.snode_size, self.snode_size])
                vnode_weights = weight_init([self.vnode_features_size, self.snode_size])
                bias = bias_init([self.snode_size])

                output_feature = list()
                for j in range(self.extracted_features):
                    full_kernel_list = list()
                    for i in range(self.input_features):
                        sub_kernel_list = list()
                        for k in range(self.orders):
                            sub_kernel = tf.matmul([snode_features[i]], kernel_weights[i][j][k] * laplacian[k])
                            sub_kernel_list.append(sub_kernel)
                        full_kernel = tf.add_n(sub_kernel_list)
                        full_kernel_list.append(full_kernel)
                    single_output_feature = tf.add_n(full_kernel_list)
                    output_feature.append(single_output_feature)
                full_output = tf.stack(output_feature, axis=0)
                full_output_features = tf.reshape(full_output, [self.extracted_features * self.snode_size])
                out = tf.matmul([full_output_features], feature_fc_weights)
                for i in range(self.vnode_features_size):
                    out += vnode_features[i] * vnode_weights[i]
                out += bias
                out = self.act(out)
                max_out=tf.reduce_max(out)
                out=tf.subtract(out,max_out)
                out_prob = tf.nn.softmax(out)
                return snode_features, vnode_features, laplacian, out_prob, out, max_out

    def train(self, s_feature, v_feature, laplacian, action, td_target):

        return self.sess.run([self.objective,self.entropy, self.optimize], feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian,
            self.selected_action: action,
            self.td_target: td_target
        })

    def predict(self, s_feature, v_feature, laplacian):
        return self.sess.run(self.action_prob, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian
        })

    def oooo(self, s_feature, v_feature, laplacian):
        return self.sess.run(self.out, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian
        })

    def get_gradients(self, s_feature, v_feature, laplacian, action, td_target):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian,
            self.selected_action: action,
            self.td_target: td_target
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_parameters)

    def set_network_params(self, input_network_parameters):
        return self.sess.run(self.set_network_parameters_op, feed_dict={
            i: d for i, d in zip(self.input_network_parameters, input_network_parameters)
        })

    def pick_action(self, action,prohibited, phase="Training"):
        a = np.arange(0, self.snode_size)
        res=np.zeros([SNODE_SIZE])
        mask=np.ones(SNODE_SIZE)
        for i in range(SNODE_SIZE):
            mask[i]=1-prohibited[i]
        temp=tf.multiply(action,mask)
        if (phase == "Training"):
            p=tf.log(temp)
            samples = tf.cast(tf.multinomial(p, 1), tf.int32).eval()
            #return a[tf.cast(samples[0][0],tf.int32)]
            res[samples]=1
            return res,samples[0][0]
        else:
            index=tf.argmax(action,axis=1).eval()
            res[index]=1
            return res,index[0][0]


class CriticNetwork(object):
    def __init__(self, sess, name, input_features, snode_size, extracted_features, vnode_features_size, orders,
                 learning_rate=0.001, act=tf.nn.relu):
        self.sess = sess
        self.name = name
        self.act = act
        self.input_features = input_features
        self.snode_size = snode_size
        self.extracted_features = extracted_features
        self.vnode_features_size = vnode_features_size
        self.orders = orders

        self.learning_rate = learning_rate
        self.td_target = tf.placeholder(tf.float32, [1])

        self.snode_features, self.vnode_features, self.laplacian, self.state_value = self.create_critic_network()

        self.network_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")

        self.input_network_parameters = []
        for param in self.network_parameters:
            self.input_network_parameters.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_parameters_op = []
        for idx, param in enumerate(self.input_network_parameters):
            self.set_network_parameters_op.append(self.network_parameters[idx].assign(param))

        self.td = tf.subtract(self.td_target, self.state_value)
        self.loss = tf.losses.mean_squared_error(self.td_target, self.state_value)
        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.critic_gradients = tf.gradients(self.loss, self.network_parameters)

    def create_critic_network(self):
        with tf.variable_scope("critic"):
            snode_features = tf.placeholder(tf.float32, [self.input_features, self.snode_size])
            vnode_features = tf.placeholder(tf.float32, [self.vnode_features_size])
            laplacian = tf.placeholder(tf.float32, [self.orders, self.snode_size, self.snode_size])

            kernel_weights = weight_init([self.input_features, self.extracted_features, self.orders])
            feature_fc_weights = weight_init([self.extracted_features * self.snode_size, self.snode_size])
            vnode_weights = weight_init([self.vnode_features_size, self.snode_size])
            bias = bias_init([self.snode_size])

            output_feature = list()
            for j in range(self.extracted_features):
                full_kernel_list = list()
                for i in range(self.input_features):
                    sub_kernel_list = list()
                    for k in range(self.orders):
                        sub_kernel = tf.matmul([snode_features[i]], kernel_weights[i][j][k] * laplacian[k])
                        sub_kernel_list.append(sub_kernel)
                    full_kernel = tf.add_n(sub_kernel_list)
                    full_kernel_list.append(full_kernel)
                single_output_feature = tf.add_n(full_kernel_list)
                output_feature.append(single_output_feature)
            full_output = tf.stack(output_feature, axis=0)
            full_output_features = tf.reshape(full_output, [self.extracted_features * self.snode_size])
            out = tf.matmul([full_output_features], feature_fc_weights)
            for i in range(self.vnode_features_size):
                out += vnode_features[i] * vnode_weights[i]
            out += bias
            state_value = tf.layers.dense(inputs=out, units=1, activation=None,
                                          kernel_initializer=tf.random_normal_initializer(0., .1),
                                          bias_initializer=tf.constant_initializer(0.1))
            state_value=tf.reshape(state_value,[1])
            return snode_features, vnode_features, laplacian, state_value

    def train(self, s_feature, v_feature, laplacian, td_target):

        return self.sess.run(self.optimize, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian,
            self.td_target: td_target
        })

    def predict(self, s_feature, v_feature, laplacian):
        return self.sess.run(self.state_value, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian
        })

    def get_td(self, s_feature, v_feature, laplacian, td_target):

        return self.sess.run(self.td, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian,
            self.td_target: td_target
        })

    def get_gradients(self, s_feature, v_feature, laplacian, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.snode_features: s_feature,
            self.vnode_features: v_feature,
            self.laplacian: laplacian,
            self.td_target: td_target
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_parameters)

    def set_network_params(self, input_network_parameters):
        return self.sess.run(self.set_network_parameters_op, feed_dict={
            i: d for i, d in zip(self.input_network_parameters, input_network_parameters)
        })


class RandomAgent(Agent):
    def __init__(self, name, environment):
        super(RandomAgent, self).__init__(name, environment)
        self.action = self.make_decision()

    def make_decision(self):
        current_VNR = self.environment.current_VNR
        current_pending_list = self.environment.pending_node_list[0]
        action_list = np.random.choice(self.action_space, len(current_pending_list), replace=False)
        return action_list

    def perform_action(self, action):
        for i in range(len(self.action)):
            res = self.environment.perform_action(self.action[i])
            if (res == False):
                return False
        return True


class NodeRankAgent(Agent):
    def __init__(self, name, environment):
        super(NodeRankAgent, self).__init__(name, environment)

    def make_decision(self):
        current_pending_list = self.environment.pending_node_list[0]
        # current_remaining_source =
        substrate_node_list = self.environment.substrate_network.node_rank("cpu_remaining")
        for i in range(len(substrate_node_list)):
            index = substrate_node_list[i]
            self.environment.check_node_embedding_availability()

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


# class ActorCriticAgent(Agent):

def master(network_parameter_queue, exp_queue):
    assert len(network_parameter_queue) == NUM_AGENT
    assert len(exp_queue) == NUM_AGENT
    with tf.Session() as sess:
        actor = ActorNetwork(sess, "actor_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS)
        critic = CriticNetwork(sess, "critic_master", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS)
        actor_parameters = actor.get_network_params()
        critic_parameters = critic.get_network_params()
        for i in range(NUM_AGENT):
            network_parameter_queue[i].put([actor_parameters, critic_parameters])
        for i in range(NUM_AGENT):
            exp = exp_queue[i].get()


def worker(network_parameter_queue, exp_queue):
    env = en.Environment("environment_worker", SNODE_SIZE, None)
    with tf.Session() as sess:
        actor = ActorNetwork(sess, "actor_worker", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES, VNODE_FEATURES_SIZE,
                             ORDERS)
        critic = CriticNetwork(sess, "critic_worker", INPUT_FEATURES, SNODE_SIZE, EXTRACTED_FEATURES,
                               VNODE_FEATURES_SIZE, ORDERS)
        actor_parameters, critic_parameters = network_parameter_queue.get()
        actor.set_network_params(actor_parameters)
        critic.set_network_params(critic_parameters)

        state = env.get_state()
        # action
        s_batch = []
        a_batch = []
        r_batch = []

    while (True):
        if (len(s_batch) == 0):
            state = env.get_state()
            s_batch.append(state)
            action_prob = actor.predict(state.s_feature, state.v_feature, LAPLACIAN_TENSOR)


start = time.clock()
with tf.Session() as sess:

    network = nt.SubstrateNetwork("sub", node_size=SNODE_SIZE, imported_graph=(G,node_attr,min,max))
    env = en.Environment("env", SNODE_SIZE,50000, imported_network=network,link_embedding_type="disjoint")
    aa = ActorNetwork(sess, "aa", input_features=INPUT_FEATURES, snode_size=SNODE_SIZE,
                      extracted_features=EXTRACTED_FEATURES, vnode_features_size=VNODE_FEATURES_SIZE, orders=ORDERS)
    cc=CriticNetwork(sess, "cc", input_features=INPUT_FEATURES, snode_size=SNODE_SIZE,
                      extracted_features=EXTRACTED_FEATURES, vnode_features_size=VNODE_FEATURES_SIZE, orders=ORDERS)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # s, v = env.get_state()
    #
    # now = time.clock()
    # print(now - start)
    # start = time.clock()
    # ss = aa.predict(s, v, LAPLACIAN_TENSOR)
    # print(ss)
    #
    # bb=aa.pick_action(ss,phase="Testing")
    # print(bb)
    now = time.clock()
    print(now - start)
    start = time.clock()
    for i in range(20):
        s, v = env.get_state()
        td=cc.predict(s,v,LAPLACIAN_TENSOR)
        action=aa.predict(s,v,LAPLACIAN_TENSOR)
        print("the action prob is:",action)
        action_one_hot,action_pick=aa.pick_action(action,env.substrate_network.attribute_list[4]["attributes"])
        # print("this is training feed:", action_pick, td)
        is_terminal,failure,reward=env.perform_action(action_pick)
        print(str("this is after perform action:"),is_terminal,failure,reward)
        # print()
        s_,v_=env.get_state()
        td_=cc.predict(s_,v_,LAPLACIAN_TENSOR)
        print(str("td initialized from critic:"), td)
        print(str("td acted from next critic:"), td_)
        td_error=td_target(td_,reward)

        new_td=cc.get_td(s,v,LAPLACIAN_TENSOR,td_error)
        print(str("td created from reward:"),new_td)
        cc.train(s, v, LAPLACIAN_TENSOR, td_error)

        #a=aa.train(s,v,LAPLACIAN_TENSOR,action_one_hot,td)
        a=aa.train(s,v,LAPLACIAN_TENSOR,action_one_hot,new_td)
        print(str("objective:"),a)
        now = time.clock()
        print(now - start)
        start = time.clock()
        ENTROPY_WEIGHT=ENTROPY_WEIGHT/10


    # bb = aa.pick_action(ss)
    # print("xxxxxxxxxxxxxxxx")
    # print(bb)
    #
    # now = time.clock()
    # print(now - start)
    # start = time.clock()
