'''
https://arxiv.org/pdf/1312.5602.pdf
The input to the neural
network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fullyconnected linear layer with a single output for each valid action. The number of valid actions varied
between 4 and 18
'''

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import gymnasium.wrappers

##debugging below
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
##debugging above

device = 'cuda'
max_actions = 3

def save_frame(array, file_name):
    obs = np.array(array)
    for i, o in enumerate(obs):
        img = Image.fromarray(o, 'L')
        img.save(f'frames/{file_name}_{i}.png')  

def translate_action(neural_network_action):
    if neural_network_action >= 1:
        return neural_network_action + 1
    return neural_network_action

def frames_to_tensor(frames):
    return frames.float().detach().to(device) / 255

class SimpleCNN(nn.Module):
    def __init__(self, device):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        nn.Flatten(start_dim=1), #probably maxpool this first
                                        nn.Linear(3136, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, max_actions))
        for m in self.network:
            if isinstance(m, nn.Conv2d):
                # Xavier/Glorot Initialization for Conv2d layers
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # He/Kaiming Initialization for Linear layers
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.network = self.network.to(device)
        


    def forward(self, x):
        
        if x.dim() == 3:  # If the input is of shape [4, 84, 84] because we're just playin
            x = x.unsqueeze(0)  # Unsqueeze to make it [1, 4, 84, 84] so we flatten correctly
        
        x = x.to(device)
        x = self.network(x)

        return x

class SimpleDQN:


    def __init__(self, rb_size=3000, batch_size = 32, gamma = 0.95, lr = 0.00001, update_target_estimator_frequency = 500, resume = False, export_model = True):

        self.rb = []
        
        self.gamma  = gamma

        self.model = SimpleCNN(device=device).to(device)
        self.estimator_model =  SimpleCNN(device=device).to(device)
        for param in self.estimator_model.parameters(): #estimator doesnt need grads
            param.requires_grad = False

        if resume:
            self.model.load_state_dict(torch.load('SimpleDQN.pth'))

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.epsilon = 1
        self.batch_size = batch_size
        self.export_model = export_model
        self.update_target_estimator_frequency = update_target_estimator_frequency
        
        self.huber_loss = torch.nn.HuberLoss()

        self.rb_max_len = rb_size

        self.rollout_rs = deque(maxlen= 100)

        self.exported_models = 0
        self.log_episode = False

    def __wrap_env_test(self, env): #no recording
        env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=2)
        env = gymnasium.wrappers.FrameStack(env, 4)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: torch.tensor(np.array(obs), dtype=torch.uint8).detach().to(device))
 
        return env
    
    def __wrap_env(self, env, video_frequency_episodes):

        env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=2)
        env = gymnasium.wrappers.FrameStack(env, 4)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: torch.tensor(np.array(obs), dtype=torch.uint8).detach().to(device))
        
        env = gymnasium.wrappers.RecordVideo(env, 'videos', episode_trigger=lambda x : x % video_frequency_episodes == 0)
        return env

    def __e_greedy(self, obs):

        if random.random() < self.epsilon:
            return random.randint(0, max_actions-1)
        
        return self.__greedy(obs)
    
    def __greedy(self, obs):
        q_values = self.model(obs)
        a =  torch.argmax(q_values, dim=1).item()
        return a
    
    def __action_batch(self, obs_tensor, actions):
        q_vals = self.model(obs_tensor)
        q_vals_of_actions = q_vals.gather(index=actions.unsqueeze(-1), dim=1)
        return q_vals_of_actions

    def __greedy_batch(self, obs_tensor, model):
        q_vals = model(obs_tensor).detach()
        q_max, _ = torch.max(q_vals, dim=1)
        return q_max 
    
    def __copy_model(self):
        self.estimator_model.load_state_dict(self.model.state_dict())
        for param in self.estimator_model.parameters():
            param.requires_grad = False
    def __run_episodes(self, env, episodes, policy):
    
        cum_r = 0
        for i in range(1, episodes+1):
            obs, _ = env.reset()
            done = False
            live_lost = True
            current_lives = 5
            
            while not done: 
                if live_lost:
                    obs, r, done, trunc, _ = env.step(1) #we're firing after losin life or start of an episode
                    live_lost = False
                a = policy(obs)
                obs, r, done, trunc, _ = env.step(translate_action(a))
                if env.unwrapped.ale.lives() < current_lives:
                    current_lives -=1
                    live_lost = True
                    r = -1 #punish dying

                done = done or trunc
                r = min(r, 1)   #clip reward
                cum_r +=r
        
        return cum_r
   
    def __calculate_baselines(self, env, episodes):
        print("Calculating baselines")
        #random action
        random_cum_r = self.__run_episodes(env, episodes, lambda _ : random.randint(0, max_actions-1))
        print("Random baseline calculated")

        #maximizing action
        max_mean_a = -10
        max_a = -1
        for a in range(max_actions):
            a_cum_r = self.__run_episodes(env, episodes, lambda _ : a)
            mean_a = a_cum_r / episodes
            if mean_a > max_mean_a:
                max_mean_a = mean_a
                max_a = a
        print("Max action baseline calculated")
        #maximizing + random action
        def policy(_):
            if random.random() < 0.2:
               return random.randint(0, max_actions-1)
            return max_a
        cum_max_a_rand = self.__run_episodes(env, episodes, policy)
        mean_max_a_rand = cum_max_a_rand / episodes
        
        print("Max action epsilon baseline calculated")

        with open('logs/baselines', 'w') as f:
            f.write(f"Random action mean r: {random_cum_r / episodes}\n")
            f.write(f"Max action mean r: {max_mean_a} \n")
            f.write(f"Max action epsilon mean r: {mean_max_a_rand}\n")
        
        print("Baselines calculated")

    def train(self, env, episodes, steps, calculate_baselines = False, epsilon_decay_steps =-1, starting_step = 1, update_frequency_steps = 4,log_interval = 100, export_interval = 100, video_frequency_episodes = 10000):
                
                n = starting_step
                end_epsilon = 0.1
                epsilon_decay_steps = steps if epsilon_decay_steps == -1 else epsilon_decay_steps
                epsilon_decay_step = (self.epsilon - end_epsilon) / epsilon_decay_steps

                env = self.__wrap_env(env, video_frequency_episodes)

                if calculate_baselines:
                    self.__calculate_baselines(env, 50)
                
                for i in range(1, episodes+1):
                    done = False

                    obs, _ = env.reset()
                    cum_r, cum_l = 0, 0
                    
                    current_lives = 5
                    live_lost = True
                    logged_heavy = False #debugging

                    while not done and n <= steps: 
                        
                        if live_lost:
                            obs, r, done, trunc, _ = env.step(1) #we're firing after losin life or start of an episode
                            live_lost = False

                        with torch.no_grad():
                            a = self.__e_greedy(frames_to_tensor(obs))
                            obs_n, r, done, trunc, _ = env.step(translate_action(a))
                            
                            if env.unwrapped.ale.lives() < current_lives:
                                current_lives -=1
                                live_lost = True
                                r = -1 #punish dying

                            done = done or trunc
                            r = min(r, 1)   #clip reward
                            
                            cum_r+=r
                            cum_l+=1
                            
                            if len(self.rb) < self.rb_max_len:
                                self.rb.append((obs, a, r, obs_n, done))
                            else:
                                indx = random.randint(0, len(self.rb)-1)
                                self.rb[indx] = (obs, a, r, obs_n, done)
                        
                        if n % update_frequency_steps ==0:
                            self.optimizer.zero_grad()
                            batch_size = min(self.batch_size, len(self.rb))
                            batch  = random.sample(self.rb, batch_size)
                            
                            obs_list, a_list, r_list, obs_n_list, done_list = zip(*batch)

                            obs_tensor = frames_to_tensor(torch.stack(obs_list)).detach().to(device)
                            a_tensor =   torch.tensor(a_list, dtype=torch.int64).detach().to(device)
                            r_tensor =   torch.tensor(r_list, dtype=torch.int64).detach().to(device)
                            obs_n_tensor = frames_to_tensor(torch.stack(obs_n_list)).detach().to(device)
                            done_tensor = torch.tensor(done_list, dtype=torch.bool).detach().to(device)
                            
                            qval_tensor = self.__action_batch(obs_tensor, a_tensor)
                            qval_n_tensor = self.__greedy_batch(obs_n_tensor, self.estimator_model).detach().to(device)
                                
                            target = torch.where(done_tensor, r_tensor, r_tensor + self.gamma * qval_n_tensor).detach()

                            loss = self.huber_loss(qval_tensor.squeeze(), target) #this shoudl take mean automatically (i hope?)
                            #loss = torch.pow(target - qval_tensor, 2)
                            #loss = torch.mean(loss)
                            loss.backward()
                            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0) #clip gradients, idk
                        
                            self.optimizer.step()
                        
                        if n % self.update_target_estimator_frequency == 0:
                            print("\ncopying model weights to estimator\n")
                            self.__copy_model()
                        

                        if self.log_episode and (not logged_heavy):
                            with torch.no_grad():
                                print("\nlogging q_vals from replay buffer\n")
                                with open('logs/q_vals.log', 'a') as f:
                                    q_vals = self.model(obs_tensor)
                                    f.write(f'episode: {i} | step: {n} \n{q_vals}\n')

                            logged_heavy = True

                        self.epsilon -=epsilon_decay_step
                        self.epsilon = max(self.epsilon, end_epsilon)

                        obs = obs_n
                        n+=1

                    self.rollout_rs.append(cum_r)
                    print(f"{i}: {n}/{steps} -> e: {self.epsilon:.3f} r: {cum_r}, l: {cum_l}, last_loss: {loss.item():.1e}")
    
                    self.log_episode = (i % log_interval == 0)

                    if self.log_episode:
                        print(f"\nLOG| mean rollout last 100: {sum(self.rollout_rs)/len(self.rollout_rs)}\n")
                    if self.export_model and i % export_interval == 0:
                        self.export()
                    if n > steps:
                        break


    def test(self, env, episodes = 10):

        self.epsilon = 0.1
        env = self.__wrap_env_test(env)
        print("training start")
        with torch.no_grad():
            for i in range(episodes):

                obs, _ = env.reset()
                done = False
                
                current_lives = 5
                live_lost = True

                cum_sum = 0
                while not done:

                    if live_lost:
                        obs, r, done, trunc, _ = env.step(1) #we're firing after losin life or start of an episode
                        live_lost = False

                    a = self.__e_greedy(frames_to_tensor(obs))
                    obs, r, done, trunc, _ = env.step(translate_action(a))
                    done = done or trunc
                    cum_sum += r

                    if env.unwrapped.ale.lives() < current_lives:
                        current_lives -=1
                        live_lost = True

                print("Cumsum = ", cum_sum)

    def export(self):
        print("\nexporting model to a file \n")
        mean_r = sum(self.rollout_rs)/len(self.rollout_rs)
        torch.save(self.model.state_dict(), f'models/{self.exported_models}_SimpleDQN_{mean_r}.pth')
        torch.save(self.model.state_dict(), f'SimpleDQN.pth') #save copy to base dir
        self.exported_models+=1

#make bookeeping 
#save and resume model
#probably adjust TD error 
