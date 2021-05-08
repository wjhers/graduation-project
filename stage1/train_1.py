
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
        
        agent = MAA2CAgent(env.observation_space,env.action_space,agent.ag_id)
        agents.append(agent)
        
        # print("agent.ag_id:",agent.ag_id)
        # print("observation_space[i]:",agent.obs_shape_n[agent.ag_id][0])
        # print("\n")
        # print("\n")
        # print("agent.actor.model.input:",agent.actor.model.input_shape)
        # print("agent.actor.model:",agent.actor.model.summary())
        # print("\n")
        # print("\n")
        # print("agent.critic.model.input:",agent.critic.model.input_shape)
        # print("agent.critic.model:",agent.critic.model.summary())
        # print("\n")
        # print("\n")
    
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

    print('Starting iterations...')

    while True:
    
        action_n = np.array([agent.action(obs) for agent, obs in zip(agents, obs_n)])
        
        action_n1 = change_action_n(action_n)

        new_obs_n, rew_n, done_n = env.step(action_n1)
            
        for agent in agents:
            agent.add_transition(obs_n[agent.ag_id],action_n[agent.ag_id],rew_n[agent.ag_id],new_obs_n[agent.ag_id],done_n[agent.ag_id])
        
        print("rew_n[0]:",rew_n[0])
        
        obs_n = new_obs_n

        print("agents[0].state.cache_files:",env.world.agents[0].state.cache_files)

        if done_n[0]:
            obs_n = env.reset()
            break
        
        for agent in agents:
            exp_v,td_error = agent.update(agents)
            # print("exp_v:",exp_v)
            # print("td_error:",td_error)


if __name__=="__main__":

    train()
    print("...")
