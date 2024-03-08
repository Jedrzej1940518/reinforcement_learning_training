
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

device = ''
total_steps = 0
avg_cum_r = deque(maxlen=200)
discounted_r = deque(maxlen=30)
discounted_r.append(5)

def bookkeeping(i, rewards):
    global avg_cum_r, total_steps, discounted_r
    avg_cum_r.append(len(rewards))

    print(f"{i}: {len(rewards)} \t /mean: {np.mean(avg_cum_r):.1f}\t")
    total_steps += len(rewards)

def calculate_returns(rewards, gamma = 0.95):
    global discounted_r

    returns = []
    prev_return = 0
    for r in reversed(rewards):
        returns.append(r + prev_return)
        prev_return += r
        prev_return *= gamma

    discounted_r.append(np.mean(returns))
    returns -= np.mean(discounted_r)
    return reversed(returns)

def train_nn(model, lr, env, episodes, cum_r_stop):
    global total_steps, device
    optimizer = torch.optim.SGD(model.parameters())
    for i in range(episodes):
        obs = env.reset()[0]
        obs = torch.tensor(obs).to(device)
        done = False
        trunc = False
        rewards = []
        probs = []
        while not (done or trunc):
            probabilities = model(obs)
            action = torch.multinomial(probabilities, num_samples=1).item()
            probs.append(probabilities[action])
            
            [obs, r, done, trunc, info]  = env.step(action)
            obs = torch.tensor(obs).to(device)
            rewards.append(r)

        returns = calculate_returns(rewards)
        
        losses = [-torch.log(prob) * ret for prob, ret in zip(probs, returns)]
        
        total_loss = sum(losses)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bookkeeping(i, rewards)

        if  np.mean(avg_cum_r) > cum_r_stop:
            return

def test_nn(model, env):
    global device

    obs = env.reset()[0]
    obs = torch.tensor(obs).to(device)
    done = False
    trunc = False
    while not (done or trunc):
        with torch.no_grad():
            probabilities = model(obs)
            action = torch.multinomial(probabilities, num_samples=1).item()
            [obs, r, done, trunc, info]  = env.step(action)
            obs = torch.tensor(obs).to(device)


def cart_pole():
    model = nn.Sequential(nn.Linear(4,6), nn.ReLU(), nn.Linear(6,2), nn.Softmax(dim=0))
    model.to(device)
    env = gym.make('CartPole-v1')
    train_nn(model, lr = 0.0001, env=env, episodes = 5000, cum_r_stop=250)    
    env = gym.make('CartPole-v1', render_mode = "human")
    test_nn(model, env)

def main():
    global device
    cuda_available = torch.cuda.is_available()
    device = "cpu" #torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")
    
    cart_pole()


main()