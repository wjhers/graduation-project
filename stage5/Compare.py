
import pandas as pd
import matplotlib as mb
import matplotlib.pyplot as plt
import numpy as np


def figureXX(mydata1,mydata2,mydata3,c1="MAA2C",c2="A2C",c3="LRU",yy="y-cdn_nums",title="",name="cdn_num_compare"):
    fig = plt.figure()
    # ax = fig.add_subplot(111) #添加axes实例
    plt.plot([i for i in range(len(mydata1))],mydata1,label=c1,linewidth=0.8)
    plt.plot([i for i in range(len(mydata2))],mydata2,label=c2,linewidth=0.8)
    plt.plot([i for i in range(len(mydata3))],mydata3,label=c3,linewidth=0.8)
    plt.xlabel('x-hours')
    plt.ylabel(yy)
    plt.title(title)
    # ax.grid(color='blue',ls='-.')
    plt.legend()
    plt.savefig("./results/COMPARE/"+title+name+'.png',bbox_inches = 'tight')
    plt.show()
    # plt.close()


LRU_cdn = pd.read_csv("./results/LRU/cdn_num_buffer.csv")
MAA2C_cdn = pd.read_csv("./results/MAA2C/CDN_num.csv")
SINGLE_cdn = pd.read_csv("./results/SINGLE/SINGLECDN_num.csv")
def figureCDN():
    figureXX(MAA2C_cdn['agent_3'],SINGLE_cdn['agent_3'],LRU_cdn['agent_3'],yy="y-cdn_nums",name="cdn_num_compare",title="agent_3")
    figureXX(MAA2C_cdn['agent_14'],SINGLE_cdn['agent_14'],LRU_cdn['agent_14'],yy="y-cdn_nums",name="cdn_num_compare",title="agent_14")
    figureXX(MAA2C_cdn['agent_20'],SINGLE_cdn['agent_20'],LRU_cdn['agent_20'],yy="y-cdn_nums",name="cdn_num_compare",title="agent_20")



LRU_hit = pd.read_csv("./results/LRU/hit_buffer.csv")
MAA2C_hit = pd.read_csv("./results/MAA2C/HitBuffer.csv")
SINGLE_hit = pd.read_csv("./results/SINGLE/SINGLEHitBuffer.csv")
def figureHit():
    figureXX(MAA2C_hit['agent_3'],SINGLE_hit['agent_3'],LRU_hit['agent_3'],yy="y-hit_ratio",name="hit_ratio_compare",title="agent_3")
    figureXX(MAA2C_hit['agent_14'],SINGLE_hit['agent_14'],LRU_hit['agent_14'],yy="y-hit_ratio",name="hit_ratio_compare",title="agent_14")
    figureXX(MAA2C_hit['agent_20'],SINGLE_hit['agent_20'],LRU_hit['agent_20'],yy="y-hit_ratio",name="hit_ratio_compare",title="agent_20")


LRU_reward = pd.read_csv("./results/LRU/reward_buffer.csv")
MAA2C_reward = pd.read_csv("./results/MAA2C/Reward.csv")
SINGLE_reward = pd.read_csv("./results/SINGLE/SINGLEReward.csv")
def figureReward():
    figureXX(MAA2C_reward['agent_3'],SINGLE_reward['agent_3'],LRU_reward['agent_3'],yy="y-reward",name="reward_compare",title="agent_3")
    figureXX(MAA2C_reward['agent_14'],SINGLE_reward['agent_14'],LRU_reward['agent_14'],yy="y-reward",name="reward_compare",title="agent_14")
    figureXX(MAA2C_reward['agent_20'],SINGLE_reward['agent_20'],LRU_reward['agent_20'],yy="y-reward",name="reward_compare",title="agent_20")



if __name__ == "__main__":
    # figureCDN()
    # figureHit()
    figureReward()


    print("...")