import numpy as np
import pygame
import random

import gymnasium as gym
from gymnasium import spaces


# Set up display
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Snake class
class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice(DIRECTIONS)
        self.last_dir = self.direction

        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):

        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.last_dir:
            return
        else:
            self.direction = point

    #returns True if snake died
    def move(self) -> bool:

        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * GRID_SIZE))), (cur[1] + (y * GRID_SIZE)))

        if (new[0] < 0 or new[0] >= WIDTH) or (new[1] < 0 or new[1] >= HEIGHT): #wall collision
            return True
        elif len(self.positions) > 2 and new in self.positions[2:]: #snake collision
            return True
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

        self.last_dir = self.direction
        return False

    def reset(self):
        self.length = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice(DIRECTIONS)

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, WHITE, r, 1)

# Food class
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE, random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE)

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, WHITE, r, 1)

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} #todo check higher fps

    def __init__(self, render_mode = None):
        
        self.observation_space = spaces.Box(0, GRID_SIZE - 1, shape=(4,), dtype=int) #snake position x,y, then food position x,y 
        self.action_space = spaces.Discrete(4) #up, down, left, right 

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

