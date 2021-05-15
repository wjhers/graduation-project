
import parameters as param
import numpy as np
from core_1 import World, Agent
from scenario import BaseScenario
import csr
'''
智能体相互独立场景,从cdn服务器获取
'''
class Scenario(BaseScenario):
    
    def make_world(self):
        world = World()
        # 设置world属性
        num_agents = param.E
        # add agents
        world.agents = [Agent(i,'agent '+str(i)) for i in range(num_agents)]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.t = 0
        for agent in world.agents:
            agent.resetagent()

    # action 是当前agent需要替换的文件
    def reward(self, agent, world, action):
        r = 0
        for i in range(param.F):
            if agent.action.c[i] == 1:
                r += self.mincost(agent,world,i,action)
        return r

    def cdn_num(self,agent,world):
        cdn_nums = sum(agent.action.c)
        return cdn_nums

    # 考虑没有命中的文件开销并且考虑需要替换的文件的开销
    def mincost(self,agent,world,f_id,action):
        d_f_e_t = world.requests.Time_t_Requests_of_Agent_i(agent.ag_id,world.t-1)[f_id]
        alpha = param.alpha
        beta = param.beta
        isreplace = 0
        # cdn命中
        l_e_j_t = param.cdn_latency(agent.ag_id)
        p_e_j_t = param.beta*param.traffic_cost_cdn
        # 是否需要加入替换开销
        for ff_id in action:
            if ff_id == f_id:
                isreplace = 1
                break
        
        cost1 = float(alpha*d_f_e_t*l_e_j_t + beta*d_f_e_t*p_e_j_t + beta*isreplace*p_e_j_t)
        return 0-cost1

    # 返回agent的观测值
    def observation(self,agent,world):
        # 近邻agent的id从小到大排列
        re = []
        if world.t >=0 and world.t <=167: 
            agent_cache_state = agent.state.cache_state
            agent_request = world.requests.Time_t_Requests_of_Agent_i(agent.ag_id,world.t)
            re.append(agent_cache_state)
            re.append(agent_request)

        re = np.asarray(re)
        return re

    # 是否结束0-167 24x7=168 未结束
    def done(self,world):
        if world.t < 0 or world.t >= 168:
            return True
        else:
            return False

if __name__=="__main__":
    # s = Scenario()
    # w = s.make_world()    
    # print(len(s.observation(w.agents[0],w)))
    # print("w.num_agents:",w.numagents())
    # for i in range(w.numagents()):
        # print(w.agents[i].ag_name)
    # w.step(np.asarray([[0,1,2,3] for _ in range(100)]))
    print("===============================================================")
    # print(s.observation(w.agents[0],w)[8:])
    # print(s.observation(w.agents[0],w)[:])
    # for i in range(w.numagents()):
        # print(w.agents[i].action.c)
    # for i in range(w.numagents()):
        # print("w.agents["+str(i)+"] reward:",s.reward(w.agents[i],w,np.asarray([0,1,2,3])))
    # for i in range(w.numagents()):
        # print(sum(w.agents[i].state.cache_state))
    print("...")

