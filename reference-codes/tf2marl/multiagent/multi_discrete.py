# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

import numpy as np

import gym
from gym.spaces import prng

class MultiDiscrete(gym.Space):
    '''
    一系列不同参数的动作离散空间组成多离散动作空间
    适应离散或连续动作空间
    核心：离散化表示
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    '''
    '''
    例如：
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    '''
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])  # 各个参数数组最小值
        self.high = np.array([x[1] for x in array_of_param_array]) # 各个参数数组最大值
        self.num_discrete_space = self.low.shape[0]                # 有多少个离散空间，shape[0]表示多少行

    def sample(self):
        '''
        返回一组离散化动作值，包含不同种类的离散值
        '''
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = prng.np_random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    
    def contains(self, x):
        '''
        是否包含x这个离散值
        x代表多种离散动作的值，保证每种离散动作都在各自的范围内
        '''
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)