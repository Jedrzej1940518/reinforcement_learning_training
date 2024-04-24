from gymnasium.envs.registration import register

register(
     id="environments/SnakeEnv-v0",
     entry_point="environments.envs:SnakeEnv",
     max_episode_steps=3000,
)

register(
     id="environments/ShootingGameEnv-v0",
     entry_point="environments.envs:ShootingGameEnv",
     max_episode_steps=20000,
)