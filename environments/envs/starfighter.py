
import math
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

WIDTH, HEIGHT = 1000, 1000
ASTEROIDS_NUM = 4


"""
Starfighter env - we're a space fighter trying to survive as long as possible and kill as many asteroids as possible
4 asteroids are present on the map with varying sizes and speed

observation space:
fighter:
-x,y positions
-speed magnitude
-speed direction
-current rotation

-shot cooldown
-hp left

4 x asteroids:
-x,y positions
-size
-speed magnitude
-speed direction
-hp left (numbers of shots to kill)

#potentially relative position to our ship

#later add

10 shots 
-x, y positions
-speed magnitude
-speed direction


action space:
rotate -> 0, 1, 2 -> no_rotation, left, right
shoot -> 0, 1 -> no shoot, shoot
accelerate -> 0...1 -> acceleration force (percentage)
acceleration dir -> 0...6.28 -> direction of acceleration (radians)

"""
class StarfighterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} #todo check higher fps

    def __init__(self, render_mode = None):
        fighter_max_speed = 10
        fighter_cooldown = 10
        fighetr_max_hp = 10
        fighter_high = [WIDTH, HEIGHT, fighter_max_speed, 2*math.pi, 2*math.pi, fighter_cooldown, fighetr_max_hp]

        asteroid_max_size = 20
        asteroid_max_speed = 8
        asteroid_max_hp = 3
        asteroid_high = [WIDTH, HEIGHT, asteroid_max_size, asteroid_max_speed, 2*math.pi, asteroid_max_hp]

        max_asteroids = 4
        high = fighter_high + asteroid_high * max_asteroids
        high = np.array(high).astype(np.float32)

        self.observation_space = spaces.Box(0, high, shape=(len(high),), dtype=float) 

        self.action_space = spaces.Box(0, high, shape=(len(high),), dtype=float) 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        s_pos = self.snake.get_head_position()
        f_pos = self.food.position
        return np.array([s_pos[0], s_pos[1], f_pos[0], f_pos[1]]) // GRID_SIZE

    def _get_info(self):
        return {"Info" : 0}
    
    def _action_into_direction(self, action):
        return DIRECTIONS[action]
    
    def _food_eaten(self):
        return self.snake.get_head_position() == self.food.position

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = Snake()
        self.food = Food()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info
    
    def step(self, action):
       
        dir = self._action_into_direction(action)
        self.snake.turn(dir)

        terminated = self.snake.move()
        reward = 1 if self._food_eaten() else 0  
        
        while self._food_eaten(): #randomize untill food is not on snake, fix later
            self.food.randomize_position()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake Game")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill((0, 0, 0))
        self.snake.draw(canvas)
        self.food.draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
