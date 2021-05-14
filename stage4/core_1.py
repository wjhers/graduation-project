'''
定义物理实体
    智能体：边缘服务器/基站
    世  界：多个智能体/变化的用户请求
    
    不断变化的:
        用户请求

    不变的：
        智能体的位置/序号

    智能体的动作：
        对于用户t时刻的内容请求
        1 本地命中
        2 是否与邻近交互
        3 与邻近交流
        4 与cdn交流

    智能体的状态：
        1 缓存（大小、剩余、文件）

    环境/智能体状态的更新：
        1 收集智能体的状态
            缓存状态
            缓存文件
        2 收集用户请求环境的变化
        3 更新智能体的状态
            满足用户的请求的文件
            不能满足用户请求的文件
                如何确定邻近服务器???
                cdn服务器???
                缓存

'''
import getdata
import parameters as param
import numpy as np
import pandas as pd

# 用户请求
class ClientRequests(object):
    def __init__(self):
        self.IFT = getdata.get_IFT()              # 所有边缘服务器节点的请求情况
    def Time_t_Requests_of_Agent_i(self,ag_id,t):
        return self.IFT[ag_id,:,t]
# 智能体的动作
class AgentAction(object):
    def __init__(self):
        # loc-hit
        self.loc = np.zeros(param.F,dtype=np.int32)
        self.c = np.zeros(param.F,dtype=np.int32)

    def resetagentaction(self):
        self.loc = np.zeros(param.F,dtype=np.int32)
        self.c = np.zeros(param.F,dtype=np.int32)
        # self.neighbor = None
        # self.cdn = None

# 智能体的状态
class AgentState(object):
    def __init__(self):
        self.cache_capacity = param.cache_capacity               # 边缘服务器cache容量
        self.cache_state = np.zeros(param.F,dtype=np.int32)      # 边缘服务器cache缓存状态,0,1表示是否有某个文件
        self.cache_files = np.array([-1 for _ in range(param.cache_capacity)]) # 边缘服务器真正缓存的文件
        self.files = []                                          # 所有文件缓存记录
    
    def resetagentstate(self):
        self.cache_state = np.zeros(param.F,dtype=np.int32)
        self.cache_files = np.array([-1 for _ in range(param.cache_capacity)])
        self.files = [] 
    
    def cachefull(self):                                         # cache是否满了
        if self.cache_capacity == sum(self.cache_files!=-1):
            return True
        else:
            return False
    
    def iscached(self,f_id):                                     # 文件f_id是否命中
        cached = False
        if self.cache_state[f_id] == 0:
            cached = False
        elif self.cache_state[f_id] == 1:
            cached = True
        return cached
    
    def tocache(self,f_id):
        self.files.append(f_id)
        if self.cachefull():                                      # 边缘服务器cache满了,替换操作
            self._replace(f_id)
        else:
            self._add(f_id)                                       # 未满，添加操作
    
    def _replace(self,f_id):                                      # 文件f_id替换操作
        tmp = (len(self.files)-1)%self.cache_capacity
        tmp_f_id = self.cache_files[tmp]                          
        self.cache_state[tmp_f_id] = 0
        self.cache_state[f_id] = 1
        self.cache_files[tmp] = f_id
    
    def _add(self,f_id):                                          # 文件f_id添加操作
        if self.cache_state[f_id]==0:
            # tmp = sum(self.cache_files!=-1)-1
            tmp = sum(self.cache_files!=-1)
            self.cache_files[tmp]=f_id
            self.cache_state[f_id] = 1

# 智能体
class Agent(object):
    def __init__(self,ag_id,ag_name=""):
        # name 
        self.ag_name = ag_name
        # id
        self.ag_id = int(ag_id)
        # state
        self.state = AgentState()
        # action
        self.action = AgentAction()
        self.action_callback = None
    
    def resetagent(self):
        self.state.resetagentstate()
        self.action.resetagentaction()
    
# 多智能体世界
class World(object):
    def __init__(self):
        # list of agents
        self.agents = []
        # self.num_agents = self.numagents()
        # requests of clients
        self.requests = ClientRequests()
        # communication channel dimensionality
        # 沟通渠道维度
        self.dim_c = 0
        # 世界的时间点0-168
        self.t = 0
        # 命中率存储
        self.hit_buffer = []   #[[agent0,agent1,...],[agent0,agent1,...]]
        # 最终可以存在excel里  

    @property
    def entities(self):
        return self.agents
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # action_n 每个智能体需要替换的文件集合
    def step(self,action_n):
        # 收集对于每个智能体在时间点t时的用户请求
        p_client_requests = [None] * len(self.entities)
        p_client_requests = self.apply_requests(p_client_requests)
        # 整合/处理智能体状态
        Hit = self.integrate_state(p_client_requests)

        self.hit_buffer.append(Hit)
        self.t = self.t + 1
        
        for agent in self.agents:
            self.update_agent_state(agent,action_n[agent.ag_id])

    def apply_caches(self,p_cache_states,p_cache_files):
        for i,agent in enumerate(self.agents):
            p_cache_states[i] = agent.state.cache_state
            p_cache_files[i] = agent.state.cache_files
        p_cache_states = np.asarray(p_cache_states)
        p_cache_files = np.asarray(p_cache_files)
        return p_cache_states,p_cache_files

    def apply_requests(self,p_client_requests):
        for i,agent in enumerate(self.agents):
            p_client_requests[i] = self.requests.Time_t_Requests_of_Agent_i(agent.ag_id,self.t)         
        return p_client_requests

    # def integrate_state(self,p_cache_states,p_cache_files,p_client_requests):
    def integrate_state(self,p_client_requests):
        '''
        p_cache_states[i],第i个智能体缓存状态,1xparam.F,含有文件f_id：p_cache_states[i][f_id]=1,否则为0
        p_cache_files[i],第i个智能体当前缓存的文件,1x4,含有文件f_id：数组p_cache_files某一个值为f_id,否则没有
        p_client_requests[i],第i个智能体当前的请求情况,1xparam.F,p_client_requests[i][f_id]表明请求f_id的次数
        '''
        # files loc-hit of each agent
        Hit = self.localhit(p_client_requests)
        # files neighbor-hit of each agent
        self.neightborhit(p_client_requests)
        
        return Hit

    def update_agent_state(self,agent,action):
        # action = action.tolist()
        for j in action:
            if agent.action.c[j] == 1:
                agent.state.tocache(j)

    # 计算哪些文件是本地命中
    def localhit(self,p_client_requests):
        Hit = [] #[agent0,agent1,...]
        for i,agent in enumerate(self.agents):
            loc_hit = np.zeros(param.F,dtype=np.int32)
            for j in range(len(agent.state.cache_files)):
                tmp = agent.state.cache_files[j]
                # 真正已缓存的文件tmp,本地命中
                if tmp!=-1 and p_client_requests[i][tmp]!=0:
                    loc_hit[tmp]=1

            agent.action.loc = loc_hit
            
            hit = float(sum(agent.action.loc)/len(agent.state.cache_files))
            Hit.append(hit)
        
        # 返回命中率
        return Hit

    # 计算哪些文件是近邻/cdn命中
    def neightborhit(self,p_client_requests):
        for i,agent in enumerate(self.agents):
            neighbor_hit = np.zeros(param.F,dtype=np.int32)
            for j in range(len(p_client_requests[i])):
                # 本地未命中
                if p_client_requests[i][j] != 0 and agent.action.loc[j] == 0:
                    # j 文件由哪个近邻/cdn 存储
                    neighbor_hit[j] = 1

            agent.action.c = neighbor_hit

    def numagents(self):
        return len(self.agents)


class LRUWorld(object):
    # 初始化
    def __init__(self):
        self.num_agents = param.E
        self.agents = [Agent(i,'agent '+str(i)) for i in range(self.num_agents)]
        self.requests = ClientRequests()
        # 存储
        self.reward_buffer = [] #[[agent0,agent1,...],[agent0,agent1,...]]
        # 命中率存储
        self.hit_buffer = []   #[[agent0,agent1,...],[agent0,agent1,...]]
        # 请求cdn服务器的次数
        self.cdn_num_buffer = [] #[[agent0,agent1,...],[agent0,agent1,...]]      

        # 世界的时间点0-168
        self.t = 0
    
    # 重置
    def reset_lruworld(self):
        self.t = 0
        # 不重置动作、状态
        # for agent in agents:
            # agent.resetagentaction()


    def MAINProcess(self,MAXEPOISE=param.MAXEPOISE):
        for i in range(MAXEPOISE):
            for index_t in range(param.T):
                Reward,Hit,CDN_num = self.LRUReplace(index_t)
                self.reward_buffer.append(Reward)
                self.hit_buffer.append(Hit)
                self.cdn_num_buffer.append(CDN_num)
            
            self.reset_lruworld()
        
        
        # 奖励值存储
        reward_buffer = np.array(self.reward_buffer)
        # 命中率存储
        hit_buffer = np.array(self.hit_buffer)
        # 请求cdn服务器的次数
        cdn_num_buffer = np.array(self.cdn_num_buffer)
        
        ##写入文件
        pd_data1 = pd.DataFrame(reward_buffer,columns=['agent_'+str(i) for i in range(100)])
        pd_data2 = pd.DataFrame(hit_buffer,columns=['agent_'+str(i) for i in range(100)])
        pd_data3 = pd.DataFrame(cdn_num_buffer,columns=['agent_'+str(i) for i in range(100)])
        
        
        pd_data1.to_csv('./results/LRU/reward_buffer.csv')
        pd_data2.to_csv('./results/LRU/hit_buffer.csv')
        pd_data3.to_csv('./results/LRU/cdn_num_buffer.csv')
        

    # LRU执行动作
    def LRUReplace(self,index_t):
        # index_t此时的时刻
        self.t = index_t
        Reward = []  #[agent0,agent1,...]
        Hit = []     #[agent0,agent1,...]
        CDN_num = [] #[agent0,agent1,...]
        
        for agent in self.agents:
            reward,hit,cdn_num = self.Processing(agent,index_t)
            Reward.append(reward)
            Hit.append(hit)
            CDN_num.append(cdn_num)
        
        return Reward,Hit,CDN_num

    # 处理过程
    def Processing(self,agent,index_t):
        agent_id_requests = self.requests.Time_t_Requests_of_Agent_i(agent.ag_id,index_t)
        
        # 本地命中
        loc_hit = np.zeros(param.F,dtype=np.int32)
        for j in range(len(agent.state.cache_files)):
            tmp = agent.state.cache_files[j]
            # 真正已缓存的文件tmp,本地命中
            if tmp!=-1 and agent_id_requests[tmp]!=0:
                loc_hit[tmp]=1
        agent.action.loc = loc_hit
 
        # 请求cdn服务器
        neighbor_hit = np.zeros(param.F,dtype=np.int32)
        for j in range(len(agent_id_requests)):
            # 本地未命中
            if agent_id_requests[j] != 0 and agent.action.loc[j] == 0:
                # j 文件由 cdn 存储
                neighbor_hit[j] = 1
        
        agent.action.c = neighbor_hit

        # 奖励值
        alpha = param.alpha
        beta = param.beta
        l_e_j_t = param.cdn_latency(agent.ag_id)
        p_e_j_t = param.beta*param.traffic_cost_cdn
        isreplace = 1
        
        reward = 0
        for i in range(len(agent.action.c)):
            if agent.action.c[i] == 1:
                d_f_e_t = agent_id_requests[i]
                reward += float(alpha*d_f_e_t*l_e_j_t + beta*d_f_e_t*p_e_j_t + beta*isreplace*p_e_j_t)
        
        
        # 将cdn服务器内容取出来放入缓存中
        for i in range(len(agent.action.c)):
            if agent.action.c[i] == 1:
                agent.state.tocache(j) #最近最少使用替换

        reward = 0 - reward
        hit = float(sum(agent.action.loc)/len(agent.state.cache_files))
        cdn_num = sum(agent.action.c)
        # 返回奖励值、命中率、请求cdn服务器的次数
        return reward,hit,cdn_num


if __name__ =="__main__":
    # w = LRUWorld()
    # w.MAINProcess()

    print("...")
