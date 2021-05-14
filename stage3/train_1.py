
import time
import os
from typing import List

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver

from environment_1 import MultiAgentEnv
from maa2c import MAA2CAgent
import parameters as param
import getdata

if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)


def make_env(scenario_name) -> MultiAgentEnv:
    '''
    Create an environment
    :param scenario_name:
    :return:
    '''
    import scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    # create world
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.done)
    return env

def get_agents(env):
    agents = []

    for agent in env.agents:
        
        agent = MAA2CAgent(env.observation_space,env.action_space,agent.ag_id,agent.ag_name)
        agents.append(agent)
    
    return agents

def change_action_n(action_n): # [[P0,..,P99],...,[P0,...,P99]]
    ans = []
    for action in action_n:
        x = np.array(np.argsort(action)[param.F-param.cache_capacity:])
        ans.append(x)
    ans = np.array(ans)
    return ans

def train():

    env = make_env('simple_scenario')

    agents = get_agents(env)
    obs_n = env.reset()
    
    for i in range(param.MAXEPOISE):
        print('Starting'+str(i+1)+' iterations...')
        while True:
        
            action_n = np.array([agent.action(obs) for agent, obs in zip(agents, obs_n)])
            
            action_n1 = change_action_n(action_n)

            new_obs_n, rew_n, done_n = env.step(action_n1)
                
            for agent in agents:
                agent.add_transition(obs_n[agent.ag_id],action_n[agent.ag_id],rew_n[agent.ag_id],new_obs_n[agent.ag_id],done_n[agent.ag_id])
            
            # print("rew_n[0]:",rew_n[0])
            
            obs_n = new_obs_n

            # print("agents[0].state.cache_files:",env.world.agents[0].state.cache_files)
            
            # print("env.world.t:",env.world.t)

            if done_n[0]:
                obs_n = env.reset()
                break

            for agent in agents:
                exp_v,td_error = agent.update(agents)

    # 绘制奖励结果图
    for agent in agents:
        getdata.figureRewarld(agent.replaybuffer.getALLRewarld(),agent.ag_name,xx="X-Iters",yy="Y-Rewarld")

if __name__=="__main__":

    train()
    print("...")
