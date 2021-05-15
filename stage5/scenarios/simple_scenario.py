
import parameters as param
import numpy as np
from core_1 import World, Agent
from scenario import BaseScenario
import csr
'''
多智能体近邻场景
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
        # 初始化智能体状态
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
        cdn_nums = 0
        for i in range(len(agent.action.c)):
            if agent.action.c[i]==1 and self.cdn_hit(agent,world,i)==True:
                cdn_nums += 1
        return cdn_nums

    # f_id在cdn服务器上命中
    def cdn_hit(self,agent,world,f_id):
        neighbors = csr.Neighbors(agent.ag_id)
        min_n = -1
        Cost = param.MAX_FLOAT
        isnehit = False
        for n_id in neighbors:
            n_id_lochit = self.neighbors_id_loc(n_id,world)
            cost = 0
            for ii in n_id_lochit:
                if ii == f_id:
                    isnehit = True
                    cost = float(param.alpha*param.ne_latency(agent.ag_id,n_id)+param.beta*param.traffic_cost_neighbor)
                    if Cost > cost:
                        Cost = cost 
                        min_n = int(n_id)
                        break
        
        # 近邻命中
        if isnehit==True:
            return False
        else:
            return True

    # 考虑没有命中的文件开销并且考虑需要替换的文件的开销
    def mincost(self,agent,world,f_id,action):
        # 得到agent的所有邻居 ndarray
        neighbors = csr.Neighbors(agent.ag_id)
        # 提供最小开销的近邻/cdn
        # 几个近邻都一样，选择id最小的近邻
        min_n = -1
        # Cost = param.MAX_INT
        Cost = param.MAX_FLOAT
        isnehit = False
        isreplace = 0 # f_id是否需要替换
        for n_id in neighbors:
            n_id_lochit = self.neighbors_id_loc(n_id,world)
            cost = 0
            for ii in n_id_lochit:
                if ii == f_id:
                    isnehit = True
                    cost = float(param.alpha*param.ne_latency(agent.ag_id,n_id)+param.beta*param.traffic_cost_neighbor)
                    if Cost > cost:
                        Cost = cost 
                        min_n = int(n_id)
                        break
        
        d_f_e_t = world.requests.Time_t_Requests_of_Agent_i(agent.ag_id,world.t-1)[f_id]
        alpha = param.alpha
        beta = param.beta
        # 近邻命中
        if isnehit==True:
            l_e_j_t = param.ne_latency(agent.ag_id,min_n)
            p_e_j_t = param.beta*param.traffic_cost_neighbor
        else: # cdn命中
            l_e_j_t = param.cdn_latency(agent.ag_id)
            p_e_j_t = param.beta*param.traffic_cost_cdn
        
        # 是否需要加入替换开销
        for ff_id in action:
            if ff_id == f_id:
                isreplace = 1
                break
        
        # cost1 = int(alpha*d_f_e_t*l_e_j_t + beta*d_f_e_t*p_e_j_t + beta*isreplace*p_e_j_t)
        cost1 = float(alpha*d_f_e_t*l_e_j_t + beta*d_f_e_t*p_e_j_t + beta*isreplace*p_e_j_t)
        
        return 0-cost1

    # 近邻的本地命中的文件
    def neighbors_id_loc(self,neighbors_id,world):
        re = []
        for i in range(param.F):
            if world.agents[neighbors_id].action.loc[i] == 1:
                re.append(i)
        re = np.array(re)
        return re

    # 返回agent的观测值
    def observation(self,agent,world):
        # 近邻agent的id从小到大排列
        re = []
        if world.t >=0 and world.t <=167: 
            neighbors = csr.Neighbors(agent.ag_id)
            agent_cache_state = agent.state.cache_state
            agent_request = world.requests.Time_t_Requests_of_Agent_i(agent.ag_id,world.t)
            re.append(agent_cache_state)
            re.append(agent_request)
            for n_id in neighbors: # 近邻id从小到大
                re.append(world.agents[n_id].state.cache_state)
                re.append(world.requests.Time_t_Requests_of_Agent_i(n_id,world.t))

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

