from collections import deque
import itertools
import math
import gymnasium as gym
import numpy as np
from agents.McReinforceWithBaseline import McReinforceWithBaseline
from agents.McPolicyGradient import McPolicyGradient
from agents.TrueOnlineSarsa import TrueOnlineSarsa

# Create the CartPole environment

#observation space
#[0] - <0, 15> discreete

#action space
#[0-4] direction

#note: lake is slippery

def clamp(val, min, max):
    return min if val < min else max if val > max else val 

def make_feature_vector_fourier(observation, action):
    pos = observation
    pos /=15
    feature = (observation * 4 + action) / (15 * 4 + 4)
    obs = np.array([feature])
    n = 100
    k = 1
    fv_len = (n+1)**k #combinations times action number
    fv = np.empty(fv_len)
    combinations = itertools.product(range(n+1), repeat=k)
    for i, c in zip(range(fv_len + 1), combinations):
        fv[i] = math.cos(math.pi * np.dot(obs, c))

    return fv

def make_state_feature_vector_fourier(observation):
    feature = observation / 15
    obs = np.array([feature])
    n = 100
    k = 1
    fv_len = (n+1)**k #combinations times action number
    fv = np.empty(fv_len)
    combinations = itertools.product(range(n+1), repeat=k)
    for i, c in zip(range(fv_len + 1), combinations):
        fv[i] = math.cos(math.pi * np.dot(obs, c))

    return fv

def frozen_lake_test(agent, episodes = 100, avg_to_break = 400):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    max_len = 100
    avg_cum_r = deque(maxlen=max_len)
    #learning
    for i in range(episodes):
        starting_obs = env.reset()[0]
        cum_r = agent.iterate_episode(starting_obs, env.step, env.action_space)
        avg_cum_r.append(cum_r)
        print(i, f"/{episodes} -> cum_r: ", cum_r, f"avg from last {max_len} ", np.average(avg_cum_r))
        if np.average(avg_cum_r) > avg_to_break:
            break

    #exploiting
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode = "human")
    #agent.epsilon = min(agent.epsilon, 0.001)
    for i in range(10):
        starting_obs = env.reset()[0]
        agent.iterate_episode(starting_obs, env.step, env.action_space)

    env.close()  # Close the environment

def main():
    mock_obs = 1
    action = 1
    fv = make_feature_vector_fourier(mock_obs, action)
    fv_s = make_state_feature_vector_fourier(mock_obs)
    epsilon_decay = lambda e : e*(1-1/10000)
    tos = TrueOnlineSarsa(make_feature_vector_fourier, len(fv), epsilon_decay = epsilon_decay, epsilon=1, alpha=0.001)
    frozen_lake_test(tos, 1000, 0.9)

    #mc = McPolicyGradient(len(fv), make_feature_vector_fourier, alpha=0.00005)
    #frozen_lake_test(mc, 9000, 0.9)
    
    #mc_baseline = McReinforceWithBaseline(len(fv), make_feature_vector_fourier, make_state_feature_vector_fourier, len(fv_s), alpha=0.0005, w_alpha=0.001)
    #frozen_lake_test(mc_baseline, 4000, 0.9)

main()