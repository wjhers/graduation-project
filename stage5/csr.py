'''
内容相似度分析脚本
注：
    分析不同边缘服务器服务区域的内容相似度
'''

import parameters as param
import getdata
import numpy as np

def Neighbors(id,distance=1):
    '''
    计算邻居节点，返回邻居节点id，一维数组
    '''
    Ne = []
    d = distance * param.cols # 乘以列数
    if distance == 1:
        if id % 10 == 0 or id == 0:
            Ne = [id-d,id-d+1,id+1,id+d,id+d+1]

        elif id % 9 == 0 and id != 0:
            Ne = [id-d-1,id-d,id-1,id+d-1,id+d]

        else:
            Ne = [id-d-1,id-d,id-d+1,id-1,id+1,id+d-1,id+d,id+d+1]

    Ne = np.array(Ne)
    Ne = Ne[Ne>=0]
    Ne = Ne[Ne<100]
    return Ne

def FilesSelf(id,beginT=0,endT=24):
    '''
    默认特定一天为第0小时到第24小时
    id区域内用户请求数据的类型都有哪些
    '''
    F = []
    IFT = getdata.get_IFT()
    for i in range(param.F):
        if sum(IFT[id,i,:][beginT:endT]) != 0:
            F.append(i)

    F = np.array(F)
    return F
    
def R_iT(id,beginT=0,endT=24):
    IFT = getdata.get_IFT()
    Data = 0
    for j in range(param.F):
        Data += sum(IFT[id,j,:][beginT:endT])
    
    return Data

def CSR(beginT=0,endT=24,distance=1):
    '''
    默认邻居阈值为1
    默认特定一天为第0小时到第24小时
    '''
    IFT = getdata.get_IFT()
    csr = np.zeros(param.E)
    # 每个边缘服务器
    for i in range(param.E): 
        # 邻居节点数组
        Ne = Neighbors(i)
        F = FilesSelf(i)
        sum_Fk = 0
        #每个邻居节点
        for j in range(len(F)):
            # 在每个自己的访问类型
            for k in range(len(Ne)):
                # 邻居节点有j内容的请求
                if sum(IFT[Ne[k],F[j],:][beginT:endT])!=0:
                    sum_Fk += sum(IFT[i,F[j],:][beginT:endT])
                    break
        Rit = R_iT(i,beginT=beginT,endT=endT)
        csr[i] = sum_Fk / Rit
    return csr

if __name__ == "__main__":
    csr = CSR()
    print(csr)
    getdata.figureHeatmap(csr.reshape(param.cols,param.rows),"CSR_iT")
