
import gymnasium as gym
from SimpleDQN import *

def test():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "human")
    dqn = SimpleDQN(resume=True, export_model=False)
    dqn.test(env)

def train():
    env = gym.make('BreakoutNoFrameskip-v4', render_mode = "rgb_array")
    dqn = SimpleDQN(115_000, batch_size=32, gamma= 0.975, lr=0.00025, update_target_estimator_frequency=10_000, resume=False, export_model = True)
    dqn.train(env, 2_000_000, 2_000_000,update_frequency_steps=4, epsilon_decay_steps=1_000_000, starting_step = 1, video_frequency_episodes=100, log_interval=10, export_interval=100)
    

def main():
    train()
    #test()

main()

#250k / 500k


#divide training into 
#stabilization -> exploration -> exploitation

# self.grad_momentum = 0.95
# self.min_sq_grad = 0.01
# weight decaym

#push and then
#replay buffer size optimization

                
#base = ~ 9200
#lazy frames -> available 8976
#tensor -> available 6765

#maybe split replay buffer too

#backdrop - implement?
#dropout - implement / remove

#improve logging




