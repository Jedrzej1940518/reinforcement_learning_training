
#https://arxiv.org/pdf/1707.06347.pdf

#todo add entropy bonus
#todo move from discrete actions to continous action space

from collections import deque
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim.lr_scheduler as lr_scheduler

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

class Actor(nn.Module):
    def __init__(self, action_space, sequential_network : nn.Sequential):
        super(Actor, self).__init__()
        
        self.action_space = action_space

        self.network = sequential_network
        self.network.to(device)
        
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        if x.dim() == 1:  # If the input is of dim 1 because we're just playin
            x = x.unsqueeze(0)  # Unsqueeze it so we can properly select its halfs

        mean = x[:, :self.action_space]         # First half for means
        mean = F.tanh(mean)                     #normalize it
        log_std = x[:, self.action_space:]      # Second half for log standard deviations
        std = F.softplus(log_std)               # Standard deviation must be positive; use exp to enforce this, maybe SOFTPLUS instead
                                                
        debug_log(lambda :f"NEWLOG| mean {mean[:8]}\n std {std[:8]} \n log_std {log_std[:8]}\n")
        return mean, std
    
    def sample_continous_action(self, x: torch.Tensor):
        mean, std = self.forward(x)
        normal_dist = torch.distributions.Normal(mean, std)
        actions = normal_dist.sample()                          # Sample from the normal distribution
        actions = actions.squeeze()
        log_probs = normal_dist.log_prob(actions).sum(axis=-1)  # Sum log probabilities for multi-dimensional actions
        debug_log(lambda :f"NEWLOG| actions {actions[:8]}\n log_probs {log_probs[:8]} \n exp(log_probs){torch.exp(log_probs[:8])}\n")
        return actions, log_probs


class SimplePPO:

    def __init__(self, actor_network, action_space, critic_network, log_path, debug = False, debug_period = 10, target_device = 'cpu', video_interval = 1000, clip = 0.2, horizon = 2048, actor_lr = 0.0001, min_actor_lr =0.00001, critic_lr = 0.0003, min_critic_lr = 0.00003, epochs = 10, minibatch_size = 64, discount = 0.99, gae = 0.95, entropy_factor = 0.01):
        global device
        device = target_device

        self.horizon = horizon
        self.epochs = epochs
        self.minibatch_size = minibatch_size 
        self.clip = clip
        self.entropy_factor = entropy_factor

        self.actor = Actor(action_space, actor_network) 
        self.critic = Critic(critic_network, discount, gae)
        self.actor_optimizer = torch.optim.Adam(self.actor.network.parameters(), lr= actor_lr, maximize=True)
        self.critic_optimizer = torch.optim.Adam(self.critic.network.parameters(), lr = critic_lr)
        self.actor_lr_scheduler = lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9995)
        self.critic_lr_scheduler  = lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9995)
        
        self.min_critic_lr = min_critic_lr
        self.min_actor_lr = min_actor_lr

        #debugging below
        self.log_path = log_path
        self.global_step = 0
        self.global_episode = 0
        self.global_iterations = 0
        self.exported_models = 0
        self.video_interval = video_interval
        global debug_log
        self.debug = debug
        self.debug_period = debug_period
        #this take msg lambda to prevent evaluating the string if no debug flag is active
        debug_log = lambda msg_lambda, file = 'debug' : self._debug_log(msg_lambda, file)
        
        os.makedirs(f'{self.log_path}/logs', exist_ok=True)
        os.makedirs(f'{self.log_path}/videos', exist_ok=True)
        os.makedirs(f'{self.log_path}/models/critic', exist_ok=True)
        os.makedirs(f'{self.log_path}/models/actor', exist_ok=True)
        
        debug_log(lambda: "\n............................Another run happening............................\n")
        debug_log(lambda: f"device: {device}, horizon: {horizon}, epochs: {epochs}, minibatch_size: {minibatch_size}, clip: {clip}\n")
        debug_log(lambda: f"discount: {discount}, gae: {gae}, actor_lr: {actor_lr}, critic_lr: {critic_lr}\n")

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
                    acts, log_probs = self.actor.sample_continous_action(obs)
                    env_actions = acts.clamp(-1,1)
                    debug_log(lambda: f"NEWLOG| actions after sampling {acts}, probs {log_probs}\n env_actions {env_actions}\n")
    
                    self.global_step +=1
                    obs_n, r, done, trunc, info = env.step(list(env_actions))
                    terminal = done or trunc
                                        
                    observations.append(obs) 
                    actions.append(acts)
                    rewards.append(r)
                    observations_n.append(obs_n)
                    actions_probs.append(log_probs)
                    terminals.append(terminal)
                    cum_r +=r

                    obs = obs_n
                    
                    if terminal: #terminal state
                        print(f"iter: {self.global_iterations} | episode: {self.global_episode}| cum_r: {cum_r} | mean cum_rs last {cum_rs.maxlen}: {np.mean(cum_rs):.2f}")
                        obs, _ = env.reset()
                        cum_rs.append(cum_r)
                        cum_r = 0
                        self.global_episode += 1

            debug_log(lambda: f"global iteration: {self.global_iterations}, global episode {self.global_episode}\n")
           
            #prepare tensors
            with torch.no_grad():
                old_log_prob = torch.stack(actions_probs).squeeze()
                obs      = torch.stack(observations)
                obs_n = torch.stack(observations_n)
                acts = torch.stack(actions)
                rs = torch.tensor(rewards, device=device)
                terminals = torch.tensor(terminals, device=device)
                
                debug_log(lambda: f"NEWLOG| acts: {acts[:8]}\nold_prob {old_log_prob[:8]}\n")
            #calculate advantages
            
                vs = self.critic(obs).squeeze()
                vs_n = self.critic(obs_n).squeeze() #squeeze :|
                vs_n = torch.where(terminals, 0, vs_n)
                dts = rs + self.critic.discount * vs_n - vs
                advantages = torch.zeros_like(dts, device=device)
                advantages[-1] = dts[-1]
                for t in reversed(range(len(dts)-1)):
                    if terminals[t+1]:
                        advantages[t] = dts[t]  # Reset advantage at the end of the episode
                    else:
                        advantages[t] = dts[t] + self.critic.discount * self.critic.gae * advantages[t+1]


                debug_log(lambda: f"calculate advantages:\nterminals: {terminals[:8]}, \nrewards{rs[:8]}, \nvs: {vs[:8]}, \nvs_n: {vs_n[:8]}, \ndts{dts[:8]}, \nadvantages {advantages[:8]}")
                
                target_v = rs + self.critic.discount * vs_n


            for k in range(self.epochs):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                indices = torch.randint(0, self.horizon, (self.minibatch_size,), device=device)  # Efficient random sampling
                 
                #new_probs = self.actor(obs)
                mean, std =  self.actor(obs)                                #todo probaby should use only indexes
                normal_dist = torch.distributions.Normal(mean, std)
                new_log_probs = normal_dist.log_prob(acts).sum(axis=-1)  # Sum log probabilities for multi-dimensional actions

                debug_log(lambda: f"NEWLOG|epoch{k}:\n actions {acts[:8]}, old_log_probs {old_log_prob[:8]}, new_log_probs{new_log_probs[:8]}")

                ratio = torch.exp(new_log_probs[indices] - old_log_prob[indices])

                obj1 = ratio * advantages[indices]
                #negative_advantages = advantages[indices] < 0
                #obj2 = torch.where(negative_advantages, 1 - self.clip * advantages[indices], 1 + self.clip * advantages[indices])
                obj2 = torch.clamp(ratio, 1-self.clip, 1+ self.clip) * advantages[indices] 

                debug_log(lambda: f"NEWLOG| ratio{ratio[:8]}, obj1 {obj1[:8]}, obj2 {obj2[:8]}, advantages {advantages[indices][:8]}")
                #todo add entorpy bonus
                #new_probs = new_probs[indices]
                #new_probs = torch.clamp(new_probs, min=1e-8)
                #entropy = -new_probs * torch.log(new_probs)
                #entropy_scalar = torch.mean(entropy)

                #debug_log(lambda: f"entropy calc | \nnew_probs {new_probs[:8]}, \nentropy {entropy[:8]}\nentropy scalar {entropy_scalar}\n")

                a_loss = torch.mean(torch.min(obj1, obj2)) # + self.entropy_factor * entropy_scalar
                #verbose
                debug_log(lambda: f"loss components | obj1 (policy gradient): {torch.mean(obj1).item():.5f}, obj2 (clip): {torch.mean(obj2).item():.5f}\n")
                a_loss.backward()                                                 
                self.actor_optimizer.step()
                with torch.no_grad():
                    debug_log(lambda: f"critic loss calculation| critic obs\n{self.critic(obs[indices])[:8]}\ntarget_vs\n{target_v[indices][:8]}\n")      

                #c_loss = torch.mean(torch.pow(self.critic(obs[indices]) - target_v[indices], 2)) 
                c_loss = torch.nn.functional.smooth_l1_loss(self.critic(obs[indices]).squeeze(), target_v[indices])
                c_loss.backward()
                self.critic_optimizer.step()
               
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
                actor_lr, critic_lr = 0,0
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], self.min_actor_lr)
                    actor_lr = param_group['lr']
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], self.min_critic_lr)
                    critic_lr = param_group['lr']

                with torch.no_grad():
                    debug_log(lambda: f"epoch:{k}| actor_loss: {a_loss:.2f}, critic_loss {c_loss:.2f}, actor_lr {actor_lr}, critic_lr {critic_lr}\n")
                    debug_log(lambda: f"epoch:{k}| mean prob ratio: {torch.mean(ratio).item()}, mean_adv: {torch.mean(advantages).item()}, mean_target_v: {torch.mean(target_v).item()}\n")
                #debug_log(lambda: f"\nratio {ratio}\n obj1 {obj1}\n obj2 {obj2} \n a_loss {a_loss}\n c_loss {c_loss}\n")


    def _debug_log(self, msg_lambda, filename):
        if not self.debug or (self.global_iterations % self.debug_period != 0):
            return

        with torch.no_grad():
            msg = msg_lambda()

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