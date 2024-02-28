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
#[0] cart position , <-4.8, 4.8>
#[1] cart velocity, <-inf, inf>
#[2] pole angle, <-0.418 rad (-24 deg), 0.418 rad (24 deg)>
#[3] pole angular velocity <-inf, inf>

#action space
#[0] push cart to the left
#[1] push cart to the right

#note: acceleration depends on the angle of the pole as center of gravity moves

# Initialize the environment
#env = gym.make('CartPole-v1', render_mode="human")

def clamp(val, min, max):
    return min if val < min else max if val > max else val 

def bounded(val, min, max):
    return (clamp(val, min, max) + abs(min)) / (max + abs(min)) 

def combine(val, min_val, max_val, action, max_actions = 2):
    #(observation * 4 + action) / (15 * 4 + 4)
    val = bounded(val, min_val, max_val)
    return (val * max_actions + action) / (1 * max_actions + max_actions) 

def make_feature_vector_fourier(observation, action):
    [pos, vel, angle, angular_vel] = observation
    pos = combine(pos, -4.8, 4.8, action) #bounded(pos, -4.8, 4.8)
    vel = combine(vel, -10, 10, action) #bounded(vel, -10, 10)
    angle = combine(angle, -0.418, 0.418, action) #bounded(angle, -0.418, 0.418)
    angular_vel = combine(angular_vel, -13, 13, action) #bounded(angular_vel, -13, 13)
    
    obs = np.array([pos, vel, angle, angular_vel])

    n = 4
    k = 4
    fv_len = (n+1)**k #combinations times action number
    fv = np.empty(fv_len)
    combinations = itertools.product(range(n+1), repeat=k)
    for i, c in zip(range(fv_len + 1), combinations):
        fv[i] = math.cos(math.pi * np.dot(obs, c))

    return fv

def make_state_feature_vector_fourier(observation):
    [pos, vel, angle, angular_vel] = observation
    pos = bounded(pos, -4.8, 4.8) 
    vel = bounded(vel, -10, 10) 
    angle = bounded(angle, -0.418, 0.418)
    angular_vel = bounded(angular_vel, -13, 13)
    obs = np.array([pos, vel, angle, angular_vel])
    
    n = 4
    k = 4
    fv_len = (n+1)**k #combinations times action number
    fv = np.empty(fv_len)
    combinations = itertools.product(range(n+1), repeat=k)
    for i, c in zip(range(fv_len + 1), combinations):
        fv[i] = math.cos(math.pi * np.dot(obs, c))

    return fv

def cart_pole_test(agent, episodes = 100):
    env = gym.make('CartPole-v1')

    max_len = episodes//20
    avg_cum_r = deque(maxlen=max_len)
    #learning
    for i in range(episodes):
        starting_obs = env.reset()[0]
        cum_r = agent.iterate_episode(starting_obs, env.step, env.action_space)
        avg_cum_r.append(cum_r)
        print(i, f"/{episodes} -> cum_r: ", cum_r, f"avg from last {max_len} ", np.average(avg_cum_r))
        if np.average(avg_cum_r) > 400:
            break

    #exploiting
    env = gym.make('CartPole-v1', render_mode = "human")
    #agent.epsilon = min(agent.epsilon, 0.001)
    for i in range(10):
        starting_obs = env.reset()[0]
        agent.iterate_episode(starting_obs, env.step, env.action_space)

    env.close()  # Close the environment

def main():
    mock_obs = [0.1, 0.2, 0.3, 0.8]
    action = 1
    fv = make_feature_vector_fourier(mock_obs, action)
    fv_s = make_state_feature_vector_fourier(mock_obs)

    #tos = TrueOnlineSarsa(make_feature_vector_fourier, len(fv), epsilon=1, alpha=0.001)

    #cart_pole_test(tos)

    #mc = McPolicyGradient(len(fv), make_feature_vector_fourier, alpha= 0.005)
    #cart_pole_test(mc, 4000)
    
    mc_baseline = McReinforceWithBaseline(len(fv), make_feature_vector_fourier, make_state_feature_vector_fourier, len(fv_s), alpha=0.00025, w_alpha=0.00025, w_bias=15/650, gamma=0.95)
    cart_pole_test(mc_baseline, 4000)

main()