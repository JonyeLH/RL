"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  #表示要输出的action数目
            n_features,	#表示有多少个observation，状态和行为等
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True, #True表示可以在Terminal使用tensorboard可视化模型
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter	#表示隔了多少步后把target_net的参数更新为最新的参数
        self.memory_size = memory_size	#整个记忆库的容量
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    #使用两个相同结构的网络来实现fix Q-target
    # def _build_net(self):
    #     # ------------------ build evaluate_net ------------------
    #     self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
    #     self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
    #     with tf.variable_scope('eval_net'):
    #         # c_names(collections_names) are the collections to store variables,只是一个eval_net的参数集合
    #         # n_l1是第一层有多少神经元数目
    #         # w_initializer权重矩阵     b_initializer偏置
    #         c_names, n_l1, w_initializer, b_initializer = \
    #             ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
    #             tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
    #
    #         # first layer. collections is used later when assign to target net
    #         with tf.variable_scope('l1'):
    #             w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
    #             b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names) #
    #             l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
    #
    #         # second layer. collections is used later when assign to target net
    #         with tf.variable_scope('l2'):
    #             w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
    #             b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
    #             self.q_eval = tf.matmul(l1, w2) + b2
    #
    #         # #try build net
    #         # time_step = 20
    #         # rnn_unit = 10
    #         # batch_size = 60
    #         # input_size = 1
    #         # output_size = 1
    #         # lr = 0.0006
    #         # weights = {
    #         #     'in': tf.Variable(tf.random_normal(input_size,rnn_unit)),
    #         #     'out': tf.Variable(tf.random_normal(rnn_unit, 1))
    #         # }
    #         # biases = {
    #         #     'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    #         #     'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    #         # }
    #         # w_in=weights['in']
    #         # b_in=biases['in']
    #         # input = tf.reshape(self.n_features, [-1, self.n_features])#将tensor转成2维
    #         # input_rnn = tf.matmul(input, w_in) + b_in
    #         # input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])#再转成3维
    #         # cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    #         # init_state = cell.zero_state(batch, dtype=tf.float32)
    #         # with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
    #         #     output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,dtype=tf.float32)
    #         # output = tf.reshape(output_rnn, [-1, rnn_unit])
    #         # w_out = weights['out']
    #         # b_out = biases['out']
    #         # self.q_eval = tf.matmul(output, w_out) + b_out
    #
    #
    #     with tf.variable_scope('loss'):
    #         self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
    #     with tf.variable_scope('train'):
    #         self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    #
    #     # ------------------ build target_net ------------------
    #     self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
    #     with tf.variable_scope('target_net'):
    #         # c_names(collections_names) are the collections to store variables
    #         c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
    #
    #         # first layer. collections is used later when assign to target net
    #         with tf.variable_scope('l1'):
    #             w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
    #             b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
    #             l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
    #
    #         # second layer. collections is used later when assign to target net
    #         with tf.variable_scope('l2'):
    #             w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
    #             b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
    #             self.q_next = tf.matmul(l1, w2) + b2

    def _build_net(self):
        time_step = 20
        rnn_unit = 10
        batch_size = 60
        input_size = 1
        output_size = 1
        lr = 0.0006

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],name='Q_target')  # q_target for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_in_initializer, b_in_initializer, w_out_initializer, b_out_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                [tf.Variable(tf.random_normal([input_size, rnn_unit]))], [tf.Variable(tf.constant(0.1))], \
                [tf.Variable(tf.random_normal([rnn_unit, 1]))], [tf.Variable(tf.constant(0.1))] #, shape=[rnn_unit, ]    , shape=[1, ]

            with tf.variable_scope('L'):
                w_in = tf.get_variable('w_in',  initializer=w_in_initializer, collections=c_names)  #[self.n_features, n_l1], tf.float32,
                b_in = tf.get_variable('b_in',  initializer=b_in_initializer, collections=c_names)  #[1, n_l1],
                input = tf.reshape(self.n_features, [-1, input_size])
                input_rnn = tf.matmul(input, w_in) + b_in
                input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
                cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
                init_state = cell.zero_state(dtype=tf.float32)
                with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
                    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
                output = tf.reshape(output_rnn, [-1, rnn_unit])
                w_out = tf.get_variable('w_out', [self.n_features, n_l1], initializer=w_out_initializer,collections=c_names)
                b_out = tf.get_variable('b_out', [1, n_l1], initializer=b_out_initializer, collections=c_names)
                self.q_eval = tf.matmul(output, w_out) + b_out

                # return pred,final_states
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('L'):
                w_in = tf.get_variable('w_in', [self.n_features, n_l1], initializer=w_in_initializer, collections=c_names)  #
                b_in = tf.get_variable('b_in', [1, n_l1], initializer=b_in_initializer, collections=c_names)  #
                self.q_next = tf.matmul(self.s_, w_in) + b_in

    #记忆库
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    #更新两个网络的环境，将eval_net的参数用来更新target_net的参数
    def _replace_target_parames(self):
        t_parames = tf.get_collection('target_net_parames')
        e_parames = tf.get_collection('eval_net_parames')
        self.sess.run([tf.assgin(t,e) for t,e in zip(t_parames,e_parames)])

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


