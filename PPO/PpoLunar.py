
import gymnasium as gym

from PPO.PpoAgent import *
from utils.calculate_baselines import calculate_baselines
import torch.nn.init as init

obs_space_lunar = 8
a_space_lunar = 2

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

#critic
default_critic_lunar =  nn.Sequential(  nn.Linear(obs_space_lunar, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))

#actor 
default_actor_lunar = nn.Sequential(    nn.Linear(obs_space_lunar, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, a_space_lunar * 2)) # times 2 because we're outputting mean and std

default_critic_lunar.apply(init_weights)
default_actor_lunar.apply(init_weights)

def train():
    env = gym.make('LunarLander-v2', continuous = True, render_mode = "rgb_array")
    ppo = SimplePPO(default_actor_lunar, a_space_lunar, default_critic_lunar, "PPO/LunarLanderContinous", debug=True, debug_period = 30, target_device='cpu', minibatch_size=64)
    ppo.train(env, 5000, export_model=True, resume=False)
    
def write_baselines():
    env = gym.make('LunarLander-v2', render_mode = "rgb_array")
    calculate_baselines(env, "PPO/LunarLander", 30)

def main(): 
    #write_baselines()
    train()
    #130 cumr with 1 less layer around 12 actor

if __name__ == "__main__":
    main()
