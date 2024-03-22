
import gymnasium as gym

from DQN.SimpleDQN import *
from utils.calculate_baselines import calculate_baselines

def test():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "human")
    dqn = SimpleDQN(resume=True, export_model=False)
    dqn.test(env)

def train():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "rgb_array")
    dqn = SimpleDQN(115_000, batch_size=32, gamma= 0.975, lr=0.00025, update_target_estimator_frequency=10_000, resume=False, export_model = True)
    dqn.train(env, 2_000_000, 2_000_000,update_frequency_steps=4, epsilon_decay_steps=1_000_000, starting_step = 1, video_frequency_episodes=100, log_interval=10, export_interval=100)
    
def write_baselines():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "rgb_array")
    calculate_baselines(env, "DQN", 30)

def main():
    #train()
    #test()
    write_baselines()

if __name__ == "__main__":
    main()
