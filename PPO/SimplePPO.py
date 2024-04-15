
#https://arxiv.org/pdf/1707.06347.pdf

from collections import deque
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import gymnasium as gym

input_size = 4
output_size = 2 #for now discrete

device = 'cpu' #todo change for cuda

debug_log = None #debug function

class Critic(nn.Module):
    def __init__(self, discount = 0.99, gae = 0.95):

        super(Critic, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1))
        self.network.to(device)

        self.discount = discount
        self.gae = gae

    def forward(self, x):
        return self.network(x)
    
    def target_v(self, r, obs_n, terminal):
        debug_log(f"rewards: {r}, v_n: {self.forward(obs_n)}, target_v {r if terminal else r + self.discount * self.forward(obs_n)}, terminal? {terminal}")
        return torch.tensor([r], device=device) if terminal else r + self.discount * self.forward(obs_n)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, output_size))
        self.network.to(device)
        
    def forward(self, x):
        
        x = self.network(x)
 #       debug_log(f"debug after forward\n {x}")
        x = F.softmax(x, dim=-1)
 #       debug_log(f"debug after softmax \n {x}")
        return x
    
#Hyperparameter Value
#Horizon (T) 2048
#Adam stepsize 3 × 10−4
#Num. epochs 10
#Minibatch size 64
#Discount (γ) 0.99
#GAE parameter (λ) 0.95
    
class SimplePPO:

    def __init__(self, log_path, debug_fun = lambda t : False, clip = 0.2, horizon = 2048, lr = 0.0001, epochs = 10, minibatch_size = 64, discount = 0.99, gae = 0.95):
        self.horizon = horizon
        self.lr = lr
        self.epochs = epochs
        self.minibatch_size = minibatch_size 
        self.discount = discount
        self.gae = gae
        self.clip = clip

        #debugging below
        self.log_path = log_path
        self.debug_fun = debug_fun
        self.global_step = 0
        self.global_episode = 0
        self.global_iterations = 0
        global debug_log
        debug_log = lambda m : self._debug_log(m)
        
        os.makedirs(f'{self.log_path}/logs', exist_ok=True)
        debug_log("\n\n............................Another run happening............................\n\n")
    
    def _wrap_env_base(self, env): #no video recording

        env = gym.wrappers.TransformObservation(env, lambda obs: torch.tensor(obs, dtype=torch.float, device=device, requires_grad=False))
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = gym.wrappers.RecordVideo(env, f'{self.log_path}/videos', episode_trigger=lambda x : x % video_interval == 0, disable_logger=True)

        return env

    #probably make this for N actors
    def train(self, env, iterations):
        
        env = self._wrap_env_base(env)

        actor = Actor() 
        critic = Critic(self.discount, self.gae)
        actor_optimizer = torch.optim.Adam(actor.network.parameters(), lr=self.lr, maximize=True)
        critic_optimizer = torch.optim.Adam(critic.network.parameters(), lr = self.lr)              #todo probably seperate lr for critic

        cum_r = 0
        cum_rs = deque(maxlen=100)
        for i in range(iterations):
            print(f"iter: {self.global_iterations}| episode: {self.global_episode}| cum_r: {cum_r} | mean cum_rs last {cum_rs.maxlen}: {np.mean(cum_rs):.2f}")
            obs, _ = env.reset()
            cum_rs.append(cum_r)
            cum_r = 0
            self.global_episode += 1
            self.global_iterations +=1

            actions = []
            actions_probs = [] 
            observations = []
            observations_n = []
            rewards = []
            terminals = []

            for t in range(self.horizon):
                with torch.no_grad():
                    actions_prob = actor(obs) 
                    debug_log(f"actions before sampling {actions_prob}")
                    a = distributions.Categorical(actions_prob).sample().item() 
                    debug_log(f"actions after sampling {a}, prob {actions_prob[a]}")
                    self.global_step +=1
                    obs_n, r, done, trunc, info = env.step(a)
                    terminal = done or trunc
                    
                    observations.append(obs)
                    actions.append(a)
                    rewards.append(r)
                    observations_n.append(obs_n)
                    actions_probs.append(actions_prob[a])
                    terminals.append(terminal)
                    cum_r +=r

                    obs = obs_n
                    
                    if terminal:               #terminal state
                        print(f"iter: {self.global_iterations} | episode: {self.global_episode}| cum_r: {cum_r} | mean cum_rs last {cum_rs.maxlen}: {np.mean(cum_rs):.2f}")
                        obs, _ = env.reset()
                        cum_rs.append(cum_r)
                        cum_r = 0
                        self.global_episode += 1
        
            for k in range(self.epochs):
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                indices = random.sample(range(len(actions)), self.minibatch_size)

                sample_actions  = [actions[i] for i in indices]              
                sample_old_prob = torch.tensor([actions_probs[i] for i in indices], dtype=torch.float, device=device)
                sample_obs      = torch.stack([observations[i] for i in indices])
                
                with torch.no_grad():
                    sample_target_v = torch.stack([critic.target_v(rewards[i], observations_n[i], terminals[i]) for i in indices]) 
                    sample_adv      = torch.stack([sample_target_v - critic(observations[i]) for i in indices])

                debug_log(f"advantages: {sample_adv}, target_v {sample_target_v}, v_s{critic(sample_obs)}")

                new_probs = torch.stack([actor(observations[i])[actions[i]] for i in indices])
                
                debug_log(f"minibatch: {k} \n indices {indices}\n actions {sample_actions} \n old probs {sample_old_prob}\n new probs {new_probs}")

                ratio = new_probs/sample_old_prob
                obj1 = ratio * sample_adv
                obj2 = torch.clamp(ratio, 1-self.clip, 1+ self.clip) * sample_adv 

                a_loss = torch.mean(torch.min(obj1, obj2))
                a_loss.backward()                                                  #todo maybe add gradient clipping
                actor_optimizer.step()
                
                c_loss = torch.mean(torch.pow(critic(sample_obs) - sample_target_v, 2)) 
                c_loss.backward()
                critic_optimizer.step()
                debug_log(f"\nratio {ratio}\n obj1 {obj1}\n obj2 {obj2} \n a_loss {a_loss}\n c_loss {c_loss}\n")


    def _debug_log(self, msg, filename = 'debug'):
        if not self.debug_fun(self.global_iterations):
            return

        with open(f'{self.log_path}/logs/{filename}.log', 'a') as f:
            f.write(f'{msg}\n')



def main():
    print("Hello PPO! Debbuging....")
    debug_fun = lambda iteration : iteration % 25 == 0
    ppo = SimplePPO("PPO/CartPole/Debug", debug_fun=debug_fun, horizon=512, minibatch_size=16, epochs=4)
    env = gym.make('CartPole-v1', render_mode = "rgb_array")
    ppo.train(env, 2300)

if __name__ == "__main__":
    main()