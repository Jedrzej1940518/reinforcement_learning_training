
from environments.envs.cpp_envs.shooting_game_base.shooting_game_module import ShootingGame

import numpy as np

import gymnasium as gym
from gymnasium import spaces

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

"""
action: 
shoot (true, false)
x (0...WIDTH)
y (0...HEIGHT)

obs:
target x center (0...WIDTH)
taregt y center (0...HEIGHT)
target speed (0...20)
target radius (20...100)

"""
#sudo rm -rf build && mkdir build && cd build && cmake .. && make && mv shooting_game_module.cpython-310-x86_64-linux-gnu.so .. && cd ..
class ShootingGameEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} #todo check higher fps

    def __init__(self, render_mode = None):

        self.game = ShootingGame()
        self.global_step = 0

        low = np.array([0, 0, 0, 20]).astype(np.float32)
        high = np.array([SCREEN_WIDTH,  SCREEN_HEIGHT,  20,  100]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, shape=(len(high),), dtype=float)  
        a_low = np.array([0, 0, 0])
        a_high = np.array([1, SCREEN_WIDTH, SCREEN_HEIGHT])
        self.action_space = spaces.Box(a_low, a_high, shape=(len(a_high),), dtype=float) #up, down, left, right 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._init_render()
        #self.clock = None

    def _init_render(self):
        if self.render_mode == 'human':
            self.game.init_human_render()
        else:
            self.game.init_rgb_render()

    def _make_info(self, info):
        return {info: info}
    
    def _make_obs(self, obs):
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs, info = self.game.reset()

        if self.render_mode == "human":
            self._render_frame()

        return self._make_obs(obs), self._make_info(info)
    
    def step(self, action):
        self.global_step+=1

        observation, reward, done, trunc, info = self.game.step(action)

        if self.render_mode == "human":
            self._render_frame()

        return self._make_obs(observation), reward, done, trunc, self._make_info(info)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            self.game.render_human()

        else:
            #rbga_arr = self.game.render_rbg()
            return np.zeros([3,3,3], dtype=np.uint8)

    def close(self):
        print("add close func xd")
        #self.game.close()

