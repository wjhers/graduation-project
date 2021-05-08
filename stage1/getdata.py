'''
用户请求数据分析脚本
注：
    用户请求量的分析
    用户请求内容的分析
    
'''

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

sns.set(font_scale=1.5)
import matplotlib.pyplot as plt
import parameters as param


def get_grid():
    grid = np.loadtxt(fname="./datas/grid.csv", dtype=np.int, delimiter=",")
    return grid

def get_IFT():
    IFT = np.load(file="./datas/IFT.npy")
    IFT[IFT<10]=0                                   # 关注10次以上请求
    return IFT


IFT = get_IFT()
grid = get_grid()


def figureHeatmap(mydata,filename,xtick=2,ytick=2): # mydata二维矩阵,heatmap图
    fig = plt.figure()
    sns_plot = sns.heatmap(mydata,cmap='YlGnBu',xticklabels=xtick,yticklabels=ytick) # 步长yticklabels
    sns_plot.tick_params(labelsize=15)              # 刻度字体大小
    cax = plt.gcf().axes[-1]                        # colorbar 刻度线设置
    cax.tick_params(labelsize=15,direction='in',top='off',bottom='off',left='off',right='off')

    fig.savefig("./results/"+filename+".png")
    plt.show()

def figureCDF(mydata,filename):                     # 一维数组
    fig = plt.figure()
    ecdf = sm.distributions.ECDF(mydata)
    x = np.linspace(min(mydata),max(mydata))
    y = ecdf(x)
    plt.step(x,y)
    fig.savefig("./results/"+filename+".png")
    plt.show()


def figureFilesTimes(mydata1,mydata2,mydata3,c1="File-0",c2="File-1",c3="File-3"):
    fig = plt.figure()
    markes = ['-o','-s','-^']
    plt.plot([i for i in range(len(mydata1))],mydata1,markes[0],label=c1,linewidth=0.8)
    plt.plot([i for i in range(len(mydata2))],mydata2,markes[1],label=c2,linewidth=0.8)
    plt.plot([i for i in range(len(mydata3))],mydata3,markes[2],label=c3,linewidth=0.8)
    plt.xlabel('Hours')
    plt.ylabel('Veiwing times')
    plt.legend()
    plt.savefig("./results/"+'Veiwing-times.png')
    plt.show()

# 计算100个区域，param.F种请求内容，在周一的总数
def sumRequests(begin=0,end=24,day="Monday"):
    Monday = np.zeros(100,dtype=np.int32)
    for i in range(param.E):
        for j in range(param.F):
            Monday[i] += sum(IFT[i,j,:][begin:end])

    Monday = Monday.reshape(10,-1) #10x10
    figureHeatmap(Monday,day)
    print(Monday)

def hourRequests():
    # 每个区域每小时的各种请求内容的总数
    # E = 100  T = 168
    Edge_T = np.zeros((param.E,param.T),dtype=np.int32)
    for i in range(param.E):
        for j in range(param.T):
            Edge_T[i,j] += sum(IFT[i,:,j])

    figureHeatmap(Edge_T,"Edge_T",xtick=50,ytick=20)


if __name__ == "__main__":

    # print(grid.shape)
    # print(grid)
    # print(IFT.shape)
    # print(IFT[0,0,:])

    sumRequests()

    hourRequests()

    # 计算特定一天每种内容的均值和方差
    mean_F = np.zeros(param.F)
    std_F = np.zeros(param.F)
    
    for j in range(param.F):
        Monday_data = []
        for i in range(param.E):
            for k in range(24):
                Monday_data.append(IFT[i,j,k])
        
        mean_F[j] = sum(Monday_data)/len(Monday_data)
        std_F[j] = np.std(Monday_data)

    print(mean_F)
    print(std_F)

    ratio_F = std_F / mean_F
    print(ratio_F)
    figureCDF(ratio_F,"ratio_F")

    # 0 1 2 三种内容在一周中个小时的请求数
    F0 = np.zeros(param.T,dtype=np.int32)
    F1 = np.zeros(param.T,dtype=np.int32)
    F2 = np.zeros(param.T,dtype=np.int32)
    for j in range(param.T):
        F0[j] = sum(IFT[:,0,j])
        F1[j] = sum(IFT[:,1,j])
        F2[j] = sum(IFT[:,2,j])

    print(F0)
    figureFilesTimes(F0,F1,F2)
