

import random
import numpy as np


class TrueOnlineSarsa:
    
    def __init__(self, feature_function, dim, epsilon_decay = lambda x: x*0.995, bias = 0, gamma = 0.9, epsilon = 0.9, alpha = 0.01, decay = 0.1):
        
        # S, A -> R^d
        self.feature_function = feature_function

        
        self.dim = dim
        
        self.ws = np.full(dim, bias, dtype=np.float16)
        self.zs = np.full(dim, 0, dtype=np.float16)

        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay

        self.epsilon_base = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
    
    def __init_episode(self, state, action_space):
        self.zs.fill(0)
        self.q_old = 0 
        self.a = self.choose_action(state, action_space)
        self.x = self.feature_function(state, self.a)
    
    def choose_action(self, state, action_space):
        if random.random() < self.epsilon:
            return random.randrange(0, action_space.n)
        
        return np.argmax([np.dot(self.feature_function(state, a), self.ws) for a in range(action_space.n)])
    
    def iterate_episode(self, state, step, action_space):
        self.__init_episode(state, action_space)
        done = False
        cum_r = 0
        while not done:
            [state, reward, done, trunc, info ] = step(self.a)
            self.a = self.choose_action(state, action_space)
            x_n = self.feature_function(state, self.a)
            q = np.dot(self.ws, self.x)
            q_n = np.dot(self.ws, x_n)
            d = reward + self.gamma * q_n - q
            self.zs = self.gamma * self.decay * self.zs + (1 - self.alpha * self.gamma * self.decay* np.dot(self.zs, self.x)) * self.x
            self.ws = self.ws + self.alpha * (d + q - self.q_old) * self.zs - self.alpha * (q - self.q_old) * self.x
            self.q_old = q_n
            self.x = x_n
        
            cum_r += reward
            done = done or trunc
            self.epsilon = self.epsilon_decay(self.epsilon)
            
        
        return cum_r