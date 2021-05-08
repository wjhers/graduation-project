

import gym
from gym import spaces
import numpy as np
# from tf2marl.multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }
    def __init__(self,world,reset_callback=None,reward_callback=None,observation_callback=None,done_callback=None,info_callback=None,shared_viewer=False):
        self.world = world
        self.agents = self.world.policy_agents
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = self.world.t
        # configure spaces
        self.action_space = [] # [Discrete(100),..., Discrete(100)]
        self.observation_space = []
        for agent in self.agents:
            # action space
            # 动作空间，目前为agent的本地命中情况
            action_dim = len(agent.action.loc)
            self.action_space.append(spaces.Discrete(action_dim))
            # observation space
            # 观测空间，目前包括 agent的状态与邻近的状态（状态=cache状态，请求状态）
            obs_dim = len(observation_callback(agent, self.world))*100
            self.observation_space.append(spaces.Discrete(obs_dim))

    # action_n内容必须是整形，代表选择文件id来替换缓存
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        # info_n = {'n': []}
        
        self.agents = self.world.policy_agents
        # set action for each agent
        # for i, agent in enumerate(self.agents):
            # self._set_action(action_n[i], agent, self.action_space[i])
        
        # advance world state
        self.world.step(action_n)
        
        # record observation for each agent
        for i,agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent,action_n[i]))
            # done_n.append(self._get_done(agent))
            done_n.append(self._get_done())
            # info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
            # reward_n = [reward] * self.n
        # return obs_n, reward_n, done_n, info_n
        
        obs_n = np.asarray(obs_n)
        reward_n = np.asarray(reward_n)
        done_n = np.asarray(done_n)
        
        return obs_n, reward_n, done_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.time = self.world.t
        # reset renderer
        # 不显示窗口
        # self._reset_render()
        
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        
        obs_n = np.asarray(obs_n)
        return obs_n
        
    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world).astype(np.int32)

    # get reward for a particular agent
    def _get_reward(self,agent,action):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent,self.world,action)

    # get dones for a particular agent
    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    # get info used for benchmarking
    # def _get_info(self, agent):
        # if self.info_callback is None:
            # return {}
        # return self.info_callback(agent, self.world)