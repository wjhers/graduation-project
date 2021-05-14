
import argparse
import numpy as np
import util as util
from gym import Space
import tensorflow as tf
from gym.spaces import Box, Discrete
from replay_buffer import ReplayBuffer
from tensorflow.keras.layers import Input, Dense
import parameters as param

class MAA2CAgent(object):
    def __init__(self,obs_space_n,act_space_n,agent_index,agent_name):
    
        assert isinstance(obs_space_n[0], Space)
        assert isinstance(act_space_n[0], Space)
        
        # 由于obs_shape_n表示所有智能体的观测，但是建立智能体时只传入单个的观测值，因此，可以放到主函数中
        self.obs_shape_n = util.space_n_to_shape_n(obs_space_n) # [[800],[1200],...,[800]] ndarray
        self.act_shape_n = util.space_n_to_shape_n(act_space_n) # [[100],[100],...,[100]] ndarray
        
        self.actor = MAA2CActorNetwork(self.obs_shape_n[agent_index][0],self.act_shape_n[agent_index][0],agent_index)
        self.critic = MAA2CCriticNetwork(self.obs_shape_n[agent_index][0],agent_index)
        
        self.ag_id = agent_index
        self.ag_name = agent_name
        # 经验缓冲区
        self.replaybuffer = ReplayBuffer(80000)

    def add_transition(self,obs_n,action_n,rew_n,new_obs_n,done):
        self.replaybuffer.add(obs_n,action_n,rew_n,new_obs_n,done)

    def action(self, obs):
        # 返回的action为每个文件的概率
        return self.actor.get_action(obs)

    # 智能体参数更新
    def update(self, agents):
        assert agents[self.ag_id] is self
        # 获取数据
        obs_t, acts_t, reward, obs_t1, dones = self.replaybuffer.sample_index([self.replaybuffer._next_idx-1])
        
        obs_t, acts_t, reward, obs_t1, dones = obs_t[0], acts_t[0], reward[0], obs_t1[0], dones[0]
        
        td_error = self.critic.learn(obs_t,reward,obs_t1) # gradient = grad[r + gamma * V(s_) - V(s)]
        exp_v = self.actor.learn(obs_t,acts_t,td_error) # true_gradient = grad[logPi(s,a) * td_error]

        return exp_v,td_error

# 策略网络Actor
class MAA2CActorNetwork(object):
    def __init__(self,obs_n_shape,act_shape,agent_index,lr=param.lr_actor):
        
        self.state_dim = obs_n_shape
        self.action_dim = act_shape
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)
        self.ag_id = agent_index

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(self.state_dim/10, activation='relu'),
            Dense(self.state_dim/10, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def forward_pass(self, obs):
        x = obs.reshape(1,-1)
        # print("obs.shape:",x.shape)
        # print("model.summary:",self.model.summary())
        # print("model.input_shape:",self.model.input_shape)
        # outputs = self.model.predict(x) # ndarray
        outputs = self.model(x).numpy() # ndarray

        return outputs[0] # [P0,...,P99]

    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        return outputs

    def learn(self,obs_t,acts_t,td_error):
        with tf.GradientTape() as tape:
            acts_prob = self.model(obs_t.reshape(1,-1))
            log_prob = tf.math.log(acts_prob)
            Entropy = 0
            for i in range(len(acts_t)):
                Entropy += tf.math.log(acts_prob[0,i]) * acts_prob[0,i]
            
            exp_v = tf.reduce_mean(log_prob * td_error - param.beta_ * Entropy)
        grads = tape.gradient(exp_v, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return exp_v


class MAA2CCriticNetwork(object):
    def __init__(self,obs_n_shape,agent_index,lr=param.lr_critic):
    
        self.state_dim = obs_n_shape
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)
        self.ag_id = agent_index

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(self.state_dim/10, activation='relu'),
            Dense(self.state_dim/10, activation='relu'),
            Dense(self.state_dim/100, activation='relu'),
            Dense(1, activation='tanh')
        ])

    def learn(self,obs_t,reward,obs_t1):
        with tf.GradientTape() as tape:
            v = self.model(obs_t.reshape(1,-1))
            v_ = self.model(obs_t1.reshape(1,-1))
            td_error = reward + param.gamma * v_ - v
            loss = tf.square(td_error)  
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return td_error
