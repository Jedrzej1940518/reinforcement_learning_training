import gymnasium as gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v1", render_mode='rbg_array')

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")
env = gym.make("CartPole-v1", render_mode='rbg_array')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()