
import gymnasium as gym

from PPO.PpoAgent import *
from utils.calculate_baselines import calculate_baselines
import torch.nn.init as init

obs_space_lunar = 8
a_space_lunar = 4

class SplitActivationLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(SplitActivationLayer, self).__init__()
        # Define a linear transformation for the entire output
        self.linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        # Apply the linear transformation
        x = self.linear(x)
        # Split the tensor along the last dimension into two halves
        x1, x2 = x.chunk(2, dim=-1)
        # Apply cosine to the first half and ReLU to the second half
        x1 = torch.cos(x1)
        x2 = F.relu(x2)
        # Concatenate the two halves back together
        return torch.cat((x1, x2), dim=-1)

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
                                        nn.Linear(64, a_space_lunar), 
                                        nn.Softmax(dim=-1)) #softmax boi

default_critic_lunar.apply(init_weights)
default_actor_lunar.apply(init_weights)

def train():
    env = gym.make('LunarLander-v2', render_mode = "rgb_array")
    ppo = SimplePPO(default_actor_lunar, default_critic_lunar, "PPO/LunarLander", debug=True, target_device='cpu')
    ppo.train(env, 5000, export_model=True)
    
def write_baselines():
    env = gym.make('LunarLander-v2', render_mode = "rgb_array")
    calculate_baselines(env, "PPO/LunarLander", 30)

def main():
    #write_baselines()
    train()
    #130 cumr with 1 less layer around 12 actor

if __name__ == "__main__":
    main()
