'''
用户请求数据脚本
注：
    生成用户的请求数据，并保存
    生成服务器网格并保存
    
'''

import numpy as np
import random
import parameters as param

F = param.F # 文件内容种类F
E = param.E # 基站E 10x10网格
T = param.T # 请求总时隙T 24x7=168小时(次)
D = param.D # 用户请求某内容最大次数D



# 用户请求数据仿真与保存
# 不考虑单个用户的请求（单个用户的坐标，每一时隙请求不同种类的内容的个数），较为复杂
# 对群体请求数据进行随机初始化（每个区域、每一时隙、对不同种类的内容的请求数量），简单

# 10x10网格
def grid():
    rows = param.rows
    cols = param.cols
    grid = np.zeros((rows,cols),dtype=np.int32)


    for i in range(cols):
        for j in range(rows):
            grid[i,j] = i * 10 + j # id = i * 10 + j

    print(grid)
    np.savetxt(fname="./datas/grid.csv", X=grid, fmt="%d",delimiter=",")  # 存储为grid.csv



# 每个边缘服务器、每种内容、每个时隙 的请求数据随机模拟
# [id][F][t] = [E][F][T]

def datas():
    IFT = np.zeros((E,F,T),dtype=np.int32)
    for i in range(E):
        for j in range(F):
            for k in range(T):
            
                if k % 24 < 7:
                    IFT[i,j,k] = random.randint(0, 15)   #0-15
                elif k % 24 < 12:
                    IFT[i,j,k] = random.randint(5, 40)   #5-40
                elif k % 24 < 15:
                    IFT[i,j,k] = random.randint(8, 35)   #8-35
                elif k % 24 < 21:
                    # up = random.randint(20,50)
                    up = 50
                    IFT[i,j,k] = random.randint(0, up)   #10-50
                else:
                    IFT[i,j,k] = random.randint(0, 15)   #0-15

    print(IFT)
    np.save(file="./datas/IFT.npy", arr=IFT)                     # 存储为IFT.npy
    print(IFT.shape)


if __name__ == "__main__":
    grid()
    datas()


# 低维数组保存 <=2
# 创建数组（2维）
# a = np.arange(200).reshape((20, 10))
# 写入文件
# np.savetxt(fname="data.csv", X=a, fmt="%d",delimiter=",")
# 读取文件
# b = np.loadtxt(fname="data.csv", dtype=np.int, delimiter=",")
# 高维数组保存 >=3
# 以3维为例子
# a = np.arange(100).reshape((10, 5, 2))
# 存储
# np.save(file="data.npy", arr=a)
# 读取
# b= np.load(file="data.npy")