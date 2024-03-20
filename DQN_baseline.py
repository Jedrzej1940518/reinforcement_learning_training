from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("Breakout-v4", n_envs=1, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=1)

resume = True

if resume:
    model = DQN.load("dqn_breakout", vec_env)
else:
    model = DQN("CnnPolicy", vec_env, verbose=1, buffer_size=100000, target_update_interval=1000)

model.learn(total_timesteps=200_000, progress_bar=True)

model.save("dqn_breakout")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_breakout")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")