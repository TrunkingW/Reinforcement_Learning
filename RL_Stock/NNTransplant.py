import os
import shutil
import tensorflow as tf
import numpy as np
from tgym.envs import SpreadTrading


np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 600
MAX_EP_STEPS = 399
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 60
REPLACE_ITER_C = 60
MEMORY_CAPACITY = 5000
BATCH_SIZE = 16
VAR_MIN = 0.1
RENDER = True
LOAD = False




class Actor(object):
    def __init__(self, sess, action_dim, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        #self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.005)
            net = tf.layers.dense(s, 200, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 128, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 64, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            net = tf.layers.dense(net, 32, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                                  trainable=trainable)
            net = tf.layers.dense(net, 16, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l5',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, 
                                          name='a', trainable=trainable)
                #scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return actions

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        action = np.zeros(self.a_dim)
        s = s[np.newaxis, :]    # single state
        act_values = self.sess.run(self.a, feed_dict={S: s})[0]
        #print("===============")
        #print("act_values: ", act_values)
        #print(np.argmax(act_values))
        #print("action: ", s)
        #print("===============")
        action[np.argmax(act_values)] = 1
        #print("action: ", action)
        #print("===============")
        return action  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            #opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            opt  = tf.train.AdamOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            #self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 128, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            net = tf.layers.dense(net, 64, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                                  trainable=trainable)
            net = tf.layers.dense(net, 32, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l5',
                                  trainable=trainable)
            net = tf.layers.dense(net, 16, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l6',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

class data_group (object):
    def __init__(self, DGenerator):
        self._DGenerator = DGenerator
        self.ss = []
		
    def DGroup(self, s_):
        try:
            self.ss = self._DGenerator.next()
        except StopIteration:
            pass
        #print("selfss: ", self.ss)
        s_ = np.hstack((self.ss,s_))

        return s_
        
		
if __name__ == "__main__":

    from tgym.envs import SpreadTrading
    from tgym.gens.csvstream import CSVStreamer
    #from tgym.gens.deterministic import WavySignal
    from test import get_CSV_data

    var = 2.
    #generator = CSVStreamer(filename='./test_4.csv')
    #other_data = CSVStreamer(filename='./test_5.csv')
    generator = get_CSV_data(filename='./test_4.csv')
	#generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)    
    trading_fee = .005
    time_fee = 0
    history_length = 1

    environment = SpreadTrading(spread_coefficients=[1],
								data_generator=generator,
								trading_fee=trading_fee,
								time_fee=time_fee,
								history_length=history_length)
    #OD = data_group(other_data)
	#print("=============")
    #print("s: ", generator.next())
    #print("=============")
    #s = environment.reset()							
    #state_size = len(s)
    #action_size = len(SpreadTrading._actions)
    state_size = 6
    action_size = 3
    #print("=============")
    #print("state_size: ", state_size)
    #print("action_size: ", action_size)
    #print("=============")
	
    # all placeholder for tf
    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, state_size], name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, state_size], name='s_')
	
    sess = tf.Session()

    # Create actor and critic.
    actor = Actor(sess, action_size, LR_A, REPLACE_ITER_A)
    critic = Critic(sess, state_size, action_size, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    M = Memory(MEMORY_CAPACITY, dims=2 * state_size + action_size + 1)

    sess.run(tf.global_variables_initializer())

    for i in range(171):
        s = environment.reset()
        #s = OD.DGroup(s)
        ep_reward = 0
        #print("=============")
        #print("s: ", s)
        #print("=============")
        for j in range(3443):
            a = actor.choose_action(s)
            #print("=============")
            #print("s: ", s, " --- ", j)
            #print("=============")
            s_, r, done, _ = environment.step(a)

            #s_ = OD.DGroup(s_)
            #print("=============")
            #print("s_: ", s_, " ---- ", j)
            #print("=============")

            M.store_transition(s, a, r, s_)
	
            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_size]
                b_a = b_M[:, state_size: state_size + action_size]
                b_r = b_M[:, -state_size - 1: -state_size]
                b_s_ = b_M[:, -state_size:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)	

            s = s_
            ep_reward += r
            
            if(i>20):
                environment.render()
			
            if j == MAX_EP_STEPS-1:
            # if done:
                result = '| done' if done else '| ----'
                print('Ep:', i,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )

