
'''
数据格式转换
'''
import numpy as np
from gym.spaces import Box, Discrete
# import tensorflow as tf

def space_n_to_shape_n(space_n):
    '''
    Takes a list of gym spaces and returns a list of their shapes
    '''
    return np.array([space_to_shape(space) for space in space_n])

def space_to_shape(space):
    '''
    Takes a gym.space and returns its shape
    '''
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Unknown space type. Can't return shape.")