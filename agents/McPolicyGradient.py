

#linear soft max policy parametrization
from math import e
import random
import numpy as np


class McPolicyGradient:
    
    def __init__(self, dims, feature_vector, bias = 0, alpha = 0.0001, gamma = 0.95):
        
        #policy parameters
        self.theta = np.full(dims, bias, dtype=np.float16)
        
        #input s, a, returns a feature fector
        self.feature_vector = feature_vector
        
        self.alpha = alpha
        self.gamma = gamma
    
    def __preference(self, s, a):
        return e**np.dot(self.theta, self.feature_vector(s,a))
    
    def __return(self, episode, t):
        gamma = 1
        cum_r = 0
        for r, _, _ in episode[t:]:
            cum_r += gamma *r
            gamma*=self.gamma      
        return cum_r      
    
    def __action_probability(self, s, a, action_space):
        weights = list(map(lambda a: self.__preference(s, a), action_space))
        return self.__preference(s, a) / sum(weights)
    
    def __gradient(self, s, a, action_space):
        return self.feature_vector(s,a) - sum(map(lambda b: (self.feature_vector(s, b) * self.__action_probability(s,a, action_space)), action_space))
    
    def choose_action(self, s, action_space):
        weights = list(map(lambda a: self.__preference(s, a), action_space))
        return random.choices(action_space, weights=weights, k=1)[0]

    def iterate_episode(self, state, step, action_space):
        action_space = range(action_space.n)
        done = False
        episode = []
        while not done:
            a = self.choose_action(state, action_space)
            [next_state, r, done, trunc, info ] = step(a)
            episode.append((r, state, a))
            state = next_state
    
            done = done or trunc
        gamma = 1
        for t, (r, s, a) in enumerate(episode):
            g = self.__return(episode, t)
            self.theta += self.alpha * gamma * g * self.__gradient(s, a, action_space)
            gamma*=self.gamma
        
        return sum([r for r, s, a in episode])