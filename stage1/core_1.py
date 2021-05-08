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
        # 一维数组,本地命中了哪个文件
        # self.loc = None
        self.loc = np.zeros(param.F,dtype=np.int32)
        # cdn/neighbor to hit
        # self.c = None
        self.c = np.zeros(param.F,dtype=np.int32)
        # ask for neighbor
        # self.neighbor = None
        # ask for cdn
        # self.cdn = None
    
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
            tmp = sum(self.cache_files!=-1)-1
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
        # cannot observe the world
        # self.blind = False        
        # script behavior to execute
        self.action_callback = None
    
    def resetagent(self):
        self.state.resetagentstate()
        self.action.resetagentaction()
        # self.blind = False        
        # self.action_callback = None
    
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
        
    # return all agents in the world
    @property
    def entities(self):
        return self.agents
  
    # return all agents controllable by external policies
    # 返回所有受外部策略控制的代理
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    # 返回世界脚本控制的所有代理
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    '''
    # update state of the world
    # 更新世界状况
    def step(self):

        # # 收集智能体缓存状态
        # p_cache_states = [None] * len(self.entities)
        # # 收集智能体缓存的文件
        # p_cache_files = [None] * len(self.entities)
        # p_cache_states,p_cache_files= self.apply_caches(p_cache_states,p_cache_files)
        
        # 收集对于每个智能体在时间点t时的用户请求
        p_client_requests = [None] * len(self.entities)
        p_client_requests = self.apply_requests(p_client_requests)
        # integrate agents' state
        # 整合/处理智能体状态
        self.integrate_state(p_client_requests)
        # update agent state
        # 更新代理状态
        for agent in self.agents:
            self.update_agent_state(agent)
    '''

    # action_n 每个智能体需要替换的文件集合
    def step(self,action_n):
    
        # # 收集智能体缓存状态
        # p_cache_states = [None] * len(self.entities)
        # # 收集智能体缓存的文件
        # p_cache_files = [None] * len(self.entities)
        # p_cache_states,p_cache_files= self.apply_caches(p_cache_states,p_cache_files)
        
        # 收集对于每个智能体在时间点t时的用户请求
        p_client_requests = [None] * len(self.entities)
        p_client_requests = self.apply_requests(p_client_requests)
        # integrate agents' state
        # 整合/处理智能体状态
        self.integrate_state(p_client_requests)
        # update agent state
        # 更新代理状态
        # for agent in self.agents:
            # self.update_agent_state(agent)
        
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
            # self.t = int((self.t + 1) % param.T)
            
        return p_client_requests

    # def integrate_state(self,p_cache_states,p_cache_files,p_client_requests):
    def integrate_state(self,p_client_requests):
        '''
        p_cache_states[i],第i个智能体缓存状态,1xparam.F,含有文件f_id：p_cache_states[i][f_id]=1,否则为0
        p_cache_files[i],第i个智能体当前缓存的文件,1x4,含有文件f_id：数组p_cache_files某一个值为f_id,否则没有
        p_client_requests[i],第i个智能体当前的请求情况,1xparam.F,p_client_requests[i][f_id]表明请求f_id的次数
        '''
        # files loc-hit of each agent
        self.localhit(p_client_requests)
        # files neighbor-hit of each agent
        self.neightborhit(p_client_requests)

    # def update_agent_state(self,agent):
        # for j in range(len(agent.action.c)):
            # if agent.action.c[j] == 1:
                # agent.state.tocache(j)

    def update_agent_state(self,agent,action):
        # action = action.tolist()
        for j in action:
            if agent.action.c[j] == 1:
                agent.state.tocache(j)

    # 计算哪些文件是本地命中
    def localhit(self,p_client_requests):
        for i,agent in enumerate(self.agents):
            loc_hit = np.zeros(param.F,dtype=np.int32)
            for j in range(len(agent.state.cache_files)):
                tmp = agent.state.cache_files[j]
                # 真正已缓存的文件tmp,本地命中
                if tmp!=-1 and p_client_requests[i][tmp]!=0:
                    loc_hit[tmp]=1

            agent.action.loc = loc_hit


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



if __name__ =="__main__":
    w = World()
    print("w.num_agents:",w.numagents())
    # print(w.requests.IFT)
    for i in range(10):
        a = Agent(i)
        w.agents.append(a)

    print("w.num_agents:",w.numagents())
    w.step()
    
    for i in range(10):
    
        print("w.agents["+str(i)+"].action.loc:\n",w.agents[i].action.loc)
        
        print("w.agents["+str(i)+"].action.c:\n",sum(w.agents[i].action.c))
        
        print("w.agents["+str(i)+"].state.files:\n",w.agents[i].state.files)
        
        print("w.agents["+str(i)+"].state.cache_state:\n",w.agents[i].state.cache_state)
        
        print("w.agents["+str(i)+"].state.cache_files:\n",w.agents[i].state.cache_files)
    
        print('------------------------------------------------------------------------------')