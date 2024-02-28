
#BartoSutton page 322
#linear funciton approximation

import numpy as np

class TrueOnlineTDLamda:
    def __init__(self, dim, gamma = 0.9, decay = 0.1, alpha = 0.01, bias = 0):
        
        self.dim = dim
        
        self.ws = np.full(dim, bias, dtype=np.float16)
        self.zs = np.full(dim, 0, dtype=np.float16)

        self.decay = decay
        self.gamma = gamma
        self.alpha = alpha
        self.v_old = 0
        self.n = 1

  
    def init_episode(self):

        self.zs.fill(0)
        self.v_old = 0
        self.n = 1
    
    def learn(self, state, reward, next_state):
        alpha = self.alpha #/ self.n
        v = np.dot(self.ws, state)
        v_n = np.dot(self.ws, next_state)
        error = reward + self.gamma * v_n - v
        self.zs = self.gamma*self.decay*self.zs +  (1-alpha * self.gamma * self.decay * np.dot(self.zs, state)) * state
        self.ws = self.ws + alpha * (error + v - self.v_old) * self.zs - alpha*(v - self.v_old) * state
        self.v_old = v_n
        self.n +=1
    
    def state_value(self, state):
        return np.dot(self.ws, state)