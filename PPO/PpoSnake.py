
import math
import gymnasium as gym
import environments

from PPO.PpoAgent import *
import torch.nn.init as init
from utils.calculate_baselines import calculate_baselines

#this is performing pretty bad, probably because snake requires a lot of precision (exact move to hit food and not hit a wall)
#and this is a very probabilisitc machine

obs_space = 5 #dir, x,y, x,y
a_space = 1 # we're outputing one continous action
env_name = 'environments:environments/SnakeEnv-v0'
log_path = "PPO/Trainings/Snake"


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
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, a_space * 2)) # times 2 because we're outputting mean and std

critic_net.apply(init_weights)
actor_net.apply(init_weights)

def translate_input(net_input):
    return net_input / 600 #normalizing but probably should not divide direction by 600 :)

def translate_output(net_output):
    action = net_output.clamp(-1, 1)
    scaled_num = (action + 1) * 2
    scaled_num = math.floor(scaled_num)
    result = max(0, min(scaled_num, 3))
    return result

def train():
    env = gym.make(env_name, render_mode = "rgb_array" )
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
