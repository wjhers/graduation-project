

import gym
from gym.envs.classic_control import rendering
import numpy as np


class Test(gym.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self):
        self.viewer = rendering.Viewer(600, 400)   # 600x400 是画板的长和框
    
    # def render(self, mode='human', close=False):
        # # 下面就可以定义你要绘画的元素了
        # line1 = rendering.Line((100, 300), (500, 300))
        # line2 = rendering.Line((100, 200), (500, 200))
        # # 给元素添加颜色
        # line1.set_color(0, 0, 0)
        # line2.set_color(0, 0, 0)
        # # 把图形元素添加到画板中
        # self.viewer.add_geom(line1)
        # self.viewer.add_geom(line2)

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render(self, mode='human', close=False):

        # 画一个直径为 30 的园
        circle = rendering.make_circle(30)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(100, 200))
        # 让圆添加平移这个属性
        circle.add_attr(circle_transform)
        self.viewer.add_geom(circle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def test():
    a = 1
    b = 2
    c = 3
    import numpy as np
    aaa = np.zeros((2,2),dtype=np.int32)
    bbb = np.zeros((2,2),dtype=np.int32)
    ccc = np.zeros((2,2),dtype=np.int32)
    for i in range(2):
        for j in range(2):
            aaa[i,j] = j+1
            bbb[i,j] = 3
    print(aaa)
    print(bbb-1)
    print(aaa**2)

    # for i in range(10):
    #     if (i>0 and i <2) or (i>4 and i <10):
    #         print("fff")
    #     else:
    #         print("...") 


    print(0%10==0)

    ss = np.zeros(10,dtype=np.int32)

    ss[9] = 9

    print(ss)

    print('--------------------------------------------------------------------------------------------------------')

    acts_prob = np.zeros((1,2),dtype=np.float)
    acts_prob[0,0] = 0.8
    acts_prob[0,1] = 0.2
    print('acts_prob[0,:]:',acts_prob[0,:].shape)

    log_prob = np.log(acts_prob[0, :])
    print('log_prob:',log_prob.shape)

    Entropy = 0-sum(acts_prob[0,:]*log_prob)
    print('Entropy:',Entropy)

    print('--------------------------------------------------------------------------------------------------------')


    # p_force = [None] * len(self.entities)
    p_force = [None] * 3
    for i in range(len(p_force)):
        p_force[i] = [i,i,i]

    p_force = np.asarray(p_force)

    print(p_force)
    print(p_force.shape)


def TestZIP():
    my_list = [[1],[1,2],[1,2,3]]
    my_list = np.asarray(my_list)
    
    my_tuple = ('a','b','c')
    my_tuple = np.asarray(my_tuple)
    
    ans = [x for x in zip(my_tuple,my_list)]
    ans = np.asarray(ans)
    
    print(ans)



if __name__ == '__main__':

    # test()
    
    # t = Test()
    # while True:
        # t.render()

    # TestZIP()
    
    # x = np.array([[-1 for _ in range(4)]])
    # x = np.swapaxes(x, 0, 1)
    
    # # y = np.zeros(100,dtype=np.int32)
    
    # print(len(x))
    # from gym import Space
    # from gym import spaces
    # import util as util
    # action_dim = 100
    # x = []
    # x.append(spaces.Discrete(action_dim))
    # x.append(spaces.Discrete(action_dim*2))
    # print(x)
    # print(isinstance(x[1], Space))
    
    # x = util.space_n_to_shape_n(x)
    
    # print(x.shape) # ans=[[100],[200]]
    # print(type(x[0])) # <class 'numpy.ndarray'> ans=[100]
    # print(x[0][0]) # ans=100
    
    # x = np.array([1,2,3,4],dtype=np.int32)
    # x = x.reshape(-1,1)
    # print(x)



    # x = np.array([6,10,29,92,40,25,15,54,9,57])
    # print(x)

    # y = np.argsort(x)  # 排序后的原下标
    # print(y[10-2:])
    # # [0 8 1 6 5 2 4 7 9 3]

    # z = x[np.argsort(x)]   # 按升序访问元素返回新数组
    # print(z)
    # # array([ 5,  6, 22, 30, 34, 36, 67, 76, 84, 99])

    # m = x[sorted(np.argsort(x)[-5:])]   # 按原来的顺序返回最大的5个数
    # print(m)
    # # array([84, 67, 76, 36, 99])
    
    
    # x = 1
    # x1 = x
    # x =2
    # print(x,x1)
    
    # import random
    # num = range(0, 100)   # 范围在0到100之间，需要用到range()函数。
    # nums = random.sample(num, 20)    # 选取10个元素
    # nums = np.array(nums)
    # print(nums) # list
    
    x = ['agent '+str(i) for i in range(10)]
    print(x)
