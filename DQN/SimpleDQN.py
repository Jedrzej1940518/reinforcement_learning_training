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
import os
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
from utils.progress_bar import print_progress_bar
##debugging above

device = 'cuda'

def save_frame(array, file_name):
    obs = np.array(array)
    for i, o in enumerate(obs):
        img = Image.fromarray(o, 'L')
        img.save(f'frames/{file_name}_{i}.png')  

def frames_to_tensor(frames):
    return frames.float().detach().to(device) / 255

class SimpleCNN(nn.Module):
    def __init__(self, max_actions):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        nn.Flatten(start_dim=1), 
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


    def __init__(self, max_actions, log_path, rb_size=50_000, batch_size = 32, gamma = 0.99, lr = 0.00025, update_target_estimator_frequency = 2500, update_frequency_steps = 4, resume = False, export_model = True):
        
        self.max_actions = max_actions

        self.rb = []
        self.gamma  = gamma
        self.model = SimpleCNN(max_actions).to(device)

        if resume:
            self.model.load_state_dict(torch.load(f'{log_path}/SimpleDQN.pth'))

        self.estimator_model =  SimpleCNN(max_actions).to(device)
        self._copy_model()

        self.epsilon = 1
        self.batch_size = batch_size        
        self.rb_max_len = rb_size

        self.update_target_estimator_frequency = update_target_estimator_frequency
        self.update_frequency_steps = update_frequency_steps
        
        self.huber_loss = torch.nn.HuberLoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.log_path = log_path
        self.export_model = export_model
        self.exported_models = 0
        self.log_episode = False

        #logging dirs
        os.makedirs(f'{self.log_path}/videos', exist_ok=True)
        os.makedirs(f'{self.log_path}/frames', exist_ok=True)
        os.makedirs(f'{self.log_path}/logs', exist_ok=True)
        os.makedirs(f'{self.log_path}/models', exist_ok=True)
        
        with open(f'{self.log_path}/logs/stats.csv', 'w') as f:
            f.write("rb_size, batch_size, gamma, lr, update_target_estimator_frequency, update_frequency_steps\n")
            f.write(f'{rb_size},{batch_size},{gamma},{lr},{update_target_estimator_frequency},{update_frequency_steps}\n')
            f.write("episode,step,epsilon,recent_mean_cum_r,recent_mean_q_val\n")


    def _wrap_env_base(self, env): #no video recording
        env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=2)
        env = gymnasium.wrappers.FrameStack(env, 4)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: torch.tensor(np.array(obs), dtype=torch.uint8).detach().to(device))
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        return env
    
    def _wrap_env_video(self, env, video_interval):
        env = self._wrap_env_base(env)
        env = gymnasium.wrappers.RecordVideo(env, f'{self.log_path}/videos', episode_trigger=lambda x : x % video_interval == 0, disable_logger=True)

        return env

    def _e_greedy(self, obs):

        q_values = self.model(obs)

        if random.random() < self.epsilon:
            a =  random.randint(0, self.max_actions-1)
        else:
            a = torch.argmax(q_values, dim=1).item()
    
        return a, q_values[0][a].item() #returns action and q_val for logging purposes
    
    def _action_batch(self, obs_tensor, actions):
        q_vals = self.model(obs_tensor)
        q_vals_of_actions = q_vals.gather(index=actions.unsqueeze(-1), dim=1)
        return q_vals_of_actions

    def _greedy_batch(self, obs_tensor, model):
        q_vals = model(obs_tensor).detach()
        q_max, _ = torch.max(q_vals, dim=1)
        return q_max 
    
    def _copy_model(self):
        self.estimator_model.load_state_dict(self.model.state_dict())
        for param in self.estimator_model.parameters():
            param.requires_grad = False

    def _dummy_info(self):
        e_info = {"r": [0], "l": [1]}
        info = {"episode": e_info}
        return info

    def train(self, env, episodes, steps, epsilon_decay_steps =-1, starting_step = 1, adaptable_exploration = False, log_interval = 100, export_interval = 100, video_interval = 100):
                
                n = starting_step
                end_epsilon = 0.1
                epsilon_decay_steps = steps if epsilon_decay_steps == -1 else epsilon_decay_steps
                epsilon_decay_step = (self.epsilon - end_epsilon) / epsilon_decay_steps

                env = self._wrap_env_video(env, video_interval)
                
                q_values_recent = deque(maxlen=100)
                last_mean = 0

                for i in range(0, episodes):
                    done = False

                    obs, _ = env.reset()
                    logged_heavy = False #debugging
                    q_values = 0

                    while not done and n <= steps: 
                        
                        with torch.no_grad():
                            a, q_val = self._e_greedy(frames_to_tensor(obs))
                            obs_n, r, done, trunc, info = env.step(a)
                            done = done or trunc
                            r /=10          #normalize reward
                            r = min(r, 1)   #clip reward
                            
                            if len(self.rb) < self.rb_max_len:
                                self.rb.append((obs, a, r, obs_n, done))
                            else:
                                indx = random.randint(0, len(self.rb)-1)
                                self.rb[indx] = (obs, a, r, obs_n, done)

                            q_values+=q_val #logging
                        
                        if n % self.update_frequency_steps ==0:
                            self.optimizer.zero_grad()
                            batch_size = min(self.batch_size, len(self.rb))
                            batch  = random.sample(self.rb, batch_size)
                            
                            obs_list, a_list, r_list, obs_n_list, done_list = zip(*batch)

                            obs_tensor = frames_to_tensor(torch.stack(obs_list)).detach().to(device)
                            a_tensor =   torch.tensor(a_list, dtype=torch.int64).detach().to(device)
                            r_tensor =   torch.tensor(r_list, dtype=torch.float).detach().to(device)
                            obs_n_tensor = frames_to_tensor(torch.stack(obs_n_list)).detach().to(device)
                            done_tensor = torch.tensor(done_list, dtype=torch.bool).detach().to(device)
                            
                            qval_tensor = self._action_batch(obs_tensor, a_tensor)
                            qval_n_tensor = self._greedy_batch(obs_n_tensor, self.estimator_model).detach().to(device)
                                
                            target = torch.where(done_tensor, r_tensor, r_tensor + self.gamma * qval_n_tensor).detach()

                            loss = self.huber_loss(qval_tensor.squeeze(), target) 
                            loss.backward()                        
                            self.optimizer.step()
                        
                        if n % self.update_target_estimator_frequency == 0:
                            print("copying model weights to estimator")
                            self._copy_model()

                        if self.log_episode and (not logged_heavy):
                            with torch.no_grad():
                                print("logging q_vals from replay buffer")
                                with open(f'{self.log_path}/logs/q_vals.log', 'a') as f:
                                    q_vals_last_batch = self.model(obs_tensor)
                                    f.write(f'episode: {i} | step: {n} \n{q_vals_last_batch}\n')

                            logged_heavy = True

                        self.epsilon -=epsilon_decay_step
                        self.epsilon = max(self.epsilon, end_epsilon)

                        obs = obs_n
                        n+=1

                    if not 'episode' in info:
                        print("\nGenerating FAKE DUMMY INFO dont trust next logs :)\n")
                        info = self._dummy_info()

                    last_q_vals = q_values/ info['episode']['l'][0] #q values from last episode       
                    q_values_recent.append(last_q_vals)

                    print(f"{i}: {n}/{steps} e: {self.epsilon:.2f}, last loss: {loss.item():.1e}, cum_r: {info['episode']['r'][0]}, last mean cum_r: {last_mean:.1f}, last q_val: {last_q_vals:.1f}")

                    self.log_episode = (i % log_interval == 0)

                    if self.log_episode:
                         last_mean =  (sum(env.return_queue) / len(env.return_queue))[0] 
                         print(f"\nlast mean from {len(env.return_queue)} episodes: {last_mean:.2f}, mean q values from last {len(q_values_recent)} episodes : {sum(q_values_recent)/len(q_values_recent):.4f}\n")
                         with open(f'{self.log_path}/logs/stats.csv', 'a') as f:
                             f.write(f'{i},{n},{self.epsilon},{last_mean},{sum(q_values_recent)/len(q_values_recent)}\n')

                    if self.export_model and i % export_interval == 0:
                        self.export(last_mean)
                    if n > steps:
                        self.export(last_mean)
                        return

                    print_progress_bar(current_progress=(n/steps)*100) #fix this


    def test(self, env, episodes = 3):

        self.epsilon = 0.1
        env = self._wrap_env_base(env)

        with torch.no_grad():
            for i in range(episodes):

                obs, _ = env.reset()
                done = False

                while not done:
                    a, _ = self._e_greedy(frames_to_tensor(obs))
                    obs, r, done, trunc, info = env.step(a)
                    done = done or trunc

                print(f"{i} cum_r: ", info['episode']['r'][0])

    def export(self, mean_r):
        print(f"exporting model to a file - mean_r from last rollouts: {mean_r:.2f}")
        torch.save(self.model.state_dict(), f'{self.log_path}/models/{self.exported_models}_SimpleDQN_{mean_r:.2f}.pth')
        torch.save(self.model.state_dict(), f'{self.log_path}/SimpleDQN.pth') #save copy to base dir
        self.exported_models+=1
