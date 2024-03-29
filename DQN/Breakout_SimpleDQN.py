
import gymnasium as gym

from DQN.SimpleDQN import *
from utils.calculate_baselines import calculate_baselines

max_a_breakout = 4

def test():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "human")
    dqn = SimpleDQN(max_a_breakout, "DQN/Breakout", resume=True, export_model=False)
    dqn.test(env)

def train():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "rgb_array")
    dqn = SimpleDQN(max_a_breakout, "DQN/Breakout/Terminal_Life_Lost" , 100_000, batch_size=32, gamma= 0.99, lr=0.00001, update_target_estimator_frequency=10_000, resume=False, export_model = False)
    dqn.train(env, 2_000_000, 10_000_000, epsilon_decay_steps=1_000_000, starting_step = 1, video_interval=100, log_interval=100, export_interval=100)
    
def write_baselines():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "rgb_array")
    calculate_baselines(env, "DQN/Breakout", 30)

def main():
    #write_baselines()
    train()
    #test()

if __name__ == "__main__":
    main()
