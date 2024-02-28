

#linear soft max policy parametrization
from math import e
import random
import numpy as np


class McReinforceWithBaseline:
    
    def __init__(self, fv_dims,feature_vector, state_feature_vector, w_dim, w_alpha = 0.001, bias = 0, w_bias = 0, alpha = 0.0001, gamma = 0.95):
        
        #policy parameters
        self.theta = np.full(fv_dims, bias, dtype=np.float16)
        
        #input s, a, returns a feature fector
        self.feature_vector = feature_vector
        #input s, returns a feature vector
        self.state_feature_vector = state_feature_vector

        self.w = np.full(w_dim, w_bias, dtype=np.float16)
        
        self.w_alpha = w_alpha
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
    
    def __v(self, state):
        fv = self.state_feature_vector(state)
        return np.dot(fv, self.w)
    
    def __v_gradient(self, state):
        return self.state_feature_vector(state)
    
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
            d = g - self.__v(s)
          #  print(f"g: {g:.1f} v(s):{g-d:.1f}, d:{d:.1f}")
            self.w += self.w_alpha * d * self.__v_gradient(s)
            self.theta += self.alpha * gamma * d * self.__gradient(s, a, action_space)
            gamma*=self.gamma
        
        return sum([r for r, s, a in episode])