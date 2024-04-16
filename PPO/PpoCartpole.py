
import gymnasium as gym

from PPO.PpoAgent import *
from utils.calculate_baselines import calculate_baselines

obs_space_cartpole = 4
a_space_cartpole = 2

#critic
default_critic_cartpole = nn.Sequential(nn.Linear(obs_space_cartpole, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1))

#actor 
default_actor_cartpole = nn.Sequential(nn.Linear(obs_space_cartpole, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, a_space_cartpole), 
                                        nn.Softmax(dim=-1)) #softmax boi

def train():
    env = gym.make('CartPole-v1', render_mode = "rgb_array")
    ppo = SimplePPO(default_actor_cartpole, default_critic_cartpole, "PPO/CartPole")
    ppo.train(env, 5000, export_model=True)
    
def write_baselines():
    env = gym.make('CartPole-v1', render_mode = "rgb_array")
    calculate_baselines(env, "PPO/CartPole", 30)

def main():
    #write_baselines()
    train()

if __name__ == "__main__":
    main()
