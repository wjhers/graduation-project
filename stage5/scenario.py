import numpy as np

# defines scenario upon which the world is built
# 定义构建世界的场景
class BaseScenario(object):
    # create elements of the world
    # 创造世界元素
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    # 创造世界的初始条件
    def reset_world(self, world):
        raise NotImplementedError()
