
import math
import gymnasium as gym
import environments

from PPO.PpoAgent import *
import torch.nn.init as init
from utils.calculate_baselines import calculate_baselines

obs_space = 4 #target x, target y, target speed, target size
a_space = 3 # shoot, x, y

env_name = 'environments:environments/ShootingGameEnv-v0'
log_path = "PPO/Trainings/ShootingGame"

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

#critic
critic_net =  nn.Sequential(  nn.Linear(obs_space, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))

#actor 
actor_net = nn.Sequential(    nn.Linear(obs_space, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, a_space * 2)) # times 2 because we're outputting mean and std

critic_net.apply(init_weights)
actor_net.apply(init_weights)


#obs_space = 4 #target x, target y, target speed, target size
#a_space = 3 # shoot, x, y

def translate_input(net_input):
    new_input = [net_input[0] / 800, net_input[1] / 50, net_input[2] / 20, net_input[3] / 100] 
    return new_input #normalizing

def translate_output(net_output):
    actions = net_output.clamp(-1, 1)
    result = [actions[0] > 0, actions[1] * 800, actions[2] * 800]
    return result

def train():
    env = gym.make(env_name, render_mode = "human_render" )
    ppo = SimplePPO(actor_net, a_space, critic_net, log_path, translate_input, translate_output, debug=True, debug_period = 30, target_device='cpu', minibatch_size=64, entropy_factor=0.00001)
    ppo.train(env, 5000, export_model=True, resume=False)
    
def write_baselines():
    env = gym.make(env_name, render_mode = "rgb_array")
    calculate_baselines(env, log_path, 30)

def main(): 
    write_baselines()
    train()

if __name__ == "__main__":
    main()
