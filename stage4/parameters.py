'''
参数脚本
注：
    用户请求参数
    智能体参数
    
'''
import csr
import sys
import math

rows = int(10)     # 行
cols = int(10)     # 列
F = int(100)       # 文件内容种类F
E = int(100)       # 基站E 10x10网格
T = int(168)       # 请求总时隙T 24x7=168小时(次)
D = int(50)        # 用户请求某内容最大次数D
# cache_capacity = int(F * 0.04)  # 每个边缘服务器的cache容量
cache_capacity = int(F * 0.2)  # 每个边缘服务器的cache容量

# 平衡传输延迟与流量成本的参数
alpha = 1
beta = 2

# 近邻之间传输延迟s->ms
def ne_latency(ag_id1,neighbor_id1):
    x1 = ag_id1/10
    y1 = ag_id1%10
    x2 = neighbor_id1/10
    y2 = neighbor_id1%10
    # return int((x1-x2)**2 + (y1-y2)**2)
    return float(0.0001*(x1-x2)**2 + (y1-y2)**2)

# 与cdn服务器之间传输延迟
def cdn_latency(ag_id1):
    neighbors1 = csr.Neighbors(ag_id1)
    sumlatency = 0
    for n_id1 in neighbors1:
        sumlatency+=ne_latency(ag_id1,n_id1)
    
    # return 5*int(sumlatency/(len(neighbors1)))
    return 5*float(sumlatency/(len(neighbors1)))


# 流量成本 p_e_j
traffic_cost_neighbor = 1   
traffic_cost_cdn = 5
 
# 默认邻居数量
Neighbors = 8

# Actor-Critic网络学习率
lr_actor = 5e-4
lr_critic = 2e-4

# 折扣因子γ
gamma = 0.99

beta_ = 0.0001

# int最大值
MAX_INT=sys.maxsize

# float 最大值
MAX_FLOAT =float('inf')

# 训练迭代次数
MAXEPOISE = 2