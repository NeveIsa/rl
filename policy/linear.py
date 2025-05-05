import numpy as np
import gymnasium as gym
from scipy.special import softmax


class LinearPolicy:
    def __init__(self, env):
        self.obsv_min = env.observation_space.low.flatten()
        self.obsv_max = env.observation_space.high.flatten()
        obsv_shape = self.obsv_max.shape[0]
        #print(self.obsv_min, self.obsv_max)
        #print(obsv_shape);exit()
        if type(env.action_space) is gym.spaces.Box:
            action_shape = env.action_space.shape[0]
        else:
            print(f"not implemented action_space type: {type(env.action_space)}")
            raise

        #print(action_shape, obsv_shape)
        self.weights = np.random.randn(action_shape, obsv_shape)

    def normalize_observation(self,obsv):
        obsv = obsv.flatten()
        obsv = np.array(obsv, dtype=float)
        obsv -= self.obsv_min
        obsv /= (self.obsv_max - self.obsv_min)
        return obsv

    def act(self,obsv):
        obsv = self.normalize_observation(obsv)

        action = self.weights @ obsv
        action = softmax(action)
        #action = np.argmax(action)
        action = np.random.normal(loc=action)


        return action
