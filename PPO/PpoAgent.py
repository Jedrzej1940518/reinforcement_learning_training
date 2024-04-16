
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

device = 'cpu' #todo change for cuda
debug_log = None #debug function

class Critic(nn.Module):
    def __init__(self, sequential_network: nn.Sequential, discount = 0.99, gae = 0.95):

        super(Critic, self).__init__()
        self.network = sequential_network
        self.network.to(device)

        self.discount = discount
        self.gae = gae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def target_v(self, r, obs_n, terminal):
        v_n = self.forward(obs_n)
        v_n = v_n.squeeze()
        target_values = torch.where(terminal, r, r+self.discount*v_n)
        debug_log(f"target_v| rewards: {r[:8]}, v_n: {v_n[:8]}, target_v: {target_values[:8]}, terminal? {terminal[:8]}")
        return target_values

class Actor(nn.Module):
    def __init__(self, sequential_network : nn.Sequential):
        super(Actor, self).__init__()
        self.network = sequential_network
        self.network.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)
    
#Hyperparameter Value
#Horizon (T) 2048
#Adam stepsize 3 × 10−4
#Num. epochs 10
#Minibatch size 64
#Discount (γ) 0.99
#GAE parameter (λ) 0.95
    
class SimplePPO:

    def __init__(self, actor_network, critic_network, log_path, debug = False, target_device = 'cpu', video_interval = 1000, clip = 0.2, horizon = 2048, actor_lr = 0.0001, critic_lr = 0.0003, epochs = 10, minibatch_size = 64, discount = 0.99, gae = 0.95):
        global device
        device = target_device

        self.horizon = horizon
        self.epochs = epochs
        self.minibatch_size = minibatch_size 
        self.clip = clip

        self.actor = Actor(actor_network) 
        self.critic = Critic(critic_network, discount, gae)
        self.actor_optimizer = torch.optim.Adam(self.actor.network.parameters(), lr= actor_lr, maximize=True)
        self.critic_optimizer = torch.optim.Adam(self.critic.network.parameters(), lr = critic_lr)

        #debugging below
        self.log_path = log_path
        self.global_step = 0
        self.global_episode = 0
        self.global_iterations = 0
        self.exported_models = 0
        self.video_interval = video_interval
        global debug_log
        self.debug = debug
        debug_log = lambda msg, file = 'debug' : self._debug_log(msg, file)
        
        os.makedirs(f'{self.log_path}/logs', exist_ok=True)
        os.makedirs(f'{self.log_path}/videos', exist_ok=True)
        os.makedirs(f'{self.log_path}/models/critic', exist_ok=True)
        os.makedirs(f'{self.log_path}/models/actor', exist_ok=True)
        
        debug_log("\n............................Another run happening............................\n")
        debug_log(f"device: {device}, horizon: {horizon}, epochs: {epochs}, minibatch_size: {minibatch_size}, clip: {clip}\n")
        debug_log(f"discount: {discount}, gae: {gae}, actor_lr: {actor_lr}, critic_lr: {critic_lr}\n")

    def _wrap_env_base(self, env): #no video recording

        env = gym.wrappers.TransformObservation(env, lambda obs: torch.tensor(obs, dtype=torch.float, device=device, requires_grad=False))
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, f'{self.log_path}/videos', episode_trigger=lambda x : x % self.video_interval == 0, disable_logger=False)
        env.metadata['render_fps'] = 60
        return env

    #todo probably make this for N actors
    def train(self, env, iterations, resume = False, export_model = False, export_iteration_period = 100):
        
        env = self._wrap_env_base(env)

        if resume:
            print("Resuming training....")
            self.critic.load_state_dict(torch.load(f'{self.log_path}/Critic.pth'))
            self.actor.load_state_dict(torch.load(f'{self.log_path}/Actor.pth'))

        cum_r = 0
        cum_rs = deque(maxlen=100)
        cum_rs.append(0)

        for i in range(1, iterations):

            if export_model and (i % export_iteration_period == 0):
                self._export_model(np.mean(cum_rs))

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
                    actions_prob = self.actor(obs) 
                    #debug_log(f"actions before sampling {actions_prob}")
                    a = distributions.Categorical(actions_prob).sample().item() 
                    #debug_log(f"actions after sampling {a}, prob {actions_prob[a]}")
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
                    
                    if terminal: #terminal state
                        print(f"iter: {self.global_iterations} | episode: {self.global_episode}| cum_r: {cum_r} | mean cum_rs last {cum_rs.maxlen}: {np.mean(cum_rs):.2f}")
                        obs, _ = env.reset()
                        cum_rs.append(cum_r)
                        cum_r = 0
                        self.global_episode += 1

            debug_log(f"global iteration: {self.global_iterations}, global episode {self.global_episode}\n")

            #prepare tensors
            with torch.no_grad():
                old_prob = torch.stack(actions_probs)
                obs      = torch.stack(observations)
                obs_n = torch.stack(observations_n)
                rs = torch.tensor(rewards, device=device)
                terminals = torch.tensor(terminals, device=device)
                acts = torch.tensor(actions, device=device)
            
                target_v = self.critic.target_v(rs, obs_n, terminals)
                advantages  = target_v - self.critic(obs).squeeze()

            for k in range(self.epochs):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                indices = torch.randint(0, self.horizon, (self.minibatch_size,), device=device)  # Efficient random sampling
                 
                #debug_log(f"advantages: {sample_adv}, target_v {sample_target_v}, v_s{critic(sample_obs)}")
                new_probs = self.actor(obs)
                new_probs_actions = new_probs.gather(1, acts.unsqueeze(1)).squeeze(1)
                
                debug_log(f"probs: {new_probs[:8]}, actions {actions[:8]}, probs_actions{new_probs_actions[:8]}")

                #debug_log(f"minibatch: {k} \n indices {indices}\n actions {sample_actions} \n old probs {sample_old_prob}\n new probs {new_probs}")

                ratio = new_probs_actions[indices]/old_prob[indices]
                debug_log(f"\nsampling base tensors: indices: {indices}, new_probs: {new_probs_actions}, old_prob: {old_prob}\n")
                debug_log(f"sampled, new_probs: {new_probs_actions[indices]}, old_prob {old_prob[indices]}, ratio {ratio}")

                sampled_adv = advantages[indices]
                obj1 = ratio * sampled_adv
                obj2 = torch.clamp(ratio, 1-self.clip, 1+ self.clip) * sampled_adv 

                a_loss = torch.mean(torch.min(obj1, obj2))
                a_loss.backward()                                                  #todo maybe add gradient clipping
                self.actor_optimizer.step()
                
                c_loss = torch.mean(torch.pow(self.critic(obs[indices]) - target_v[indices], 2)) 
                c_loss.backward()
                self.critic_optimizer.step()
               
                if self.debug:
                    with torch.no_grad():
                        debug_log(f"epoch:{k}| actor_loss: {a_loss:.2f}, critic_loss {c_loss:.2f}\n")
                        debug_log(f"epoch:{k}| mean prob ratio: {torch.mean(ratio).item()}, mean_adv: {torch.mean(advantages).item()}, mean_target_v: {torch.mean(target_v).item()}\n")
                #debug_log(f"\nratio {ratio}\n obj1 {obj1}\n obj2 {obj2} \n a_loss {a_loss}\n c_loss {c_loss}\n")


    def _debug_log(self, msg, filename):
        if not self.debug:
            return

        with open(f'{self.log_path}/logs/{filename}.log', 'a') as f:
            f.write(f'{msg}\n')

    def _export_model(self, mean_r):
        print(f"exporting model to a file - mean_r from last rollouts: {mean_r:.2f}")
        torch.save(self.critic.state_dict(), f'{self.log_path}/models/critic/{self.exported_models}_Critic_{mean_r:.2f}.pth')
        torch.save(self.actor.state_dict(), f'{self.log_path}/models/actor/{self.exported_models}_Actor_{mean_r:.2f}.pth')
        torch.save(self.critic.state_dict(), f'{self.log_path}/Critic.pth')
        torch.save(self.actor.state_dict(), f'{self.log_path}/Actor.pth')
        
        with open(f'{self.log_path}/export.log', 'a') as f:
            f.write(f'exported_models: {self.exported_models}, mean_r: {mean_r}\n')

        self.exported_models+=1