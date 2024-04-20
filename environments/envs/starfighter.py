
import math
import random
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

WIDTH, HEIGHT = 1000, 1000
EDGE_L = 1000

ASTEROIDS_NUM = 4


"""
Starfighter env - we're a space fighter trying to survive as long as possible and kill as many asteroids as possible
4 asteroids are present on the map with varying sizes and velocity



4 x asteroids:

#potentially relative position to our ship

#later add

10 shots 
-x, y positions
-velocity magnitude
-velocity direction

"""

FIGHTER_MAX_VELOCITY = 10
FIGHTER_COOLDOWN  = 10
FIGHTER_MAX_HP = 10

ASTEROID_MAX_SIZE = 20
ASTEROID_MAX_VELOCITY = 8
ASTEROID_MAX_HP = 3
MAX_ASTEROIDS = 4

class Asteroid:
    def __init__(self, x, y, size, velocity, velocity_angle, hp):
        self.x = x
        self.y = y
        self.size = size
        self.velocity = velocity
        self.velocity_angle = velocity_angle
        self.max_hp = hp
        self.hp_left = hp

    def get_normalized_obs(self):
        return [self.x / WIDTH, self.y / HEIGHT, self.size / ASTEROID_MAX_SIZE, self.velocity / ASTEROID_MAX_VELOCITY, self.velocity_angle / math.pi, self.hp_left / ASTEROID_MAX_HP]

    def get_obs(self):
        """
        asteroid_num* [asteroid x, asteroid y, asteroid size, asteroid speed, asteroid speed dir, asteroid hp left]
        """
        return [self.x, self.y, self.size, self.velocity, self.velocity_angle, self.hp_left]

class Starfighter:
    def __init__(self):
        self.x = 1
        self.y = 1
        pass

    def get_normalized_obs(self):
        pass
    
    def get_obs(self):
        """
        fighter x, fighter y, fighter speed, fighter speed direction, fighter rotation, fighter cooldown, fighter hp left
        """
        pass

class StarfighterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} #todo check higher fps

    def __init__(self, render_mode = None):

        """
        fighter:
        -x,y positions -> 0...WIDTH, 0...HEIGHT
        -velocity magnitude -> 0...FIGHTER_MAX_SPEED
        -velocity direction -> -3.14....3.14
        -current rotation   -> -3.14....3.14
        -shot cooldown -> 0....FIGHTER_COOLDOWN
        -hp left -> 0....FIGHTER_MAX_HP
        """
        fighter_low = [0,0,0,-3.14,-3.14, 0, 0]
        fighter_high = [WIDTH, HEIGHT, FIGHTER_MAX_VELOCITY, math.pi, math.pi, FIGHTER_COOLDOWN, FIGHTER_MAX_HP]
        
        """
        asteroid:
        -x,y positions -> 0....WIDTH, 0....HEIGHT
        -size -> 1....ASTEROID_MAX_SIZE
        -velocity magnitude -> 1.....ASTEROID_MAX_VELOCITY
        -velocity direction -> -3.14....3.14
        -hp left (numbers of shots to kill) -> 0.....ASTEROID_MAX_HP
        """

        asteroid_low = [0,0,1, 1, -3.14, 0]
        asteroid_high = [WIDTH, HEIGHT, ASTEROID_MAX_SIZE, ASTEROID_MAX_VELOCITY, math.pi, ASTEROID_MAX_HP]

        high = fighter_high + asteroid_high * MAX_ASTEROIDS 
        high = np.array(high).astype(np.float32)
        low = fighter_low + asteroid_low * MAX_ASTEROIDS
        low = np.array(low).astype(np.float32)
        """
        obs space:
        fighter x, fighter y, fighter speed, fighter speed direction, fighter rotation, fighter cooldown, fighter hp left, + asteroid_num* [asteroid x, asteroid y, asteroid size, asteroid speed, asteroid speed dir, asteroid hp left]
        """
        self.observation_space = spaces.Box(low, high, shape=(len(high),), dtype=float) 
        
        """
        action space:
        rotate -> -1.....1 -> left, right (percentages)
        shoot -> 0...1 -> no shoot <0.5, >0.5 shoot
        accelerate -> 0...1 -> acceleration force (percentage)
        acceleration dir -> -3.14...3.14 -> direction of acceleration (radians)
        """
        action_space_low = np.array([-1,0,0,-math.pi]).astype(np.float32)
        action_space_high = np.array([1, 1, 1, math.pi]).astype(np.float32)
        self.action_space = spaces.Box(action_space_low, action_space_high, shape=(len(action_space_high),), dtype=float) 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    
    def _random_point_on_square(self):
        edge = random.randint(1, 4)
        t = random.uniform(0, EDGE_L)
        
        if edge == 1:   #top
            return t, 0
        elif edge == 2: #right
            return EDGE_L, t 
        elif edge == 3: #bottom
            return t, EDGE_L
        elif edge == 4: #left
            return 0, t
        
        return 0,0
    
    def _angle_between_points(self, ax, ay, bx, by):
        # Direction vector
        dx = bx - ax
        dy = by - ay
        return math.atan2(dy, dx) 


    def _make_asteroid(self) -> Asteroid:
        x, y = self._random_point_on_square()
        size = random.randint(1, ASTEROID_MAX_SIZE)
        velocity = random.uniform(0.1, ASTEROID_MAX_VELOCITY) #magnitude
        angle_to_starship = self._angle_between_points(x,y, self.starfighter.x, self.starfighter.y)
        velocity_dir = random.uniform(-math.pi/6, math.pi/6) + angle_to_starship #velocity in general direction of starship +- 30 degrees
        hp = random.randint(1, ASTEROID_MAX_HP)             #todo probably scale with size
        return Asteroid(x,y,size,velocity, velocity_dir, hp)

    def _get_obs(self):

        obs = self.starfighter.get_obs()
        for asteroid in self.asteroids:
            obs += asteroid.get_obs() 
        print("obs: ", obs)

        return np.array(obs)

    def _get_info(self):
        return {"Info" : 0}
    
    def _action_into_direction(self, action):
        return DIRECTIONS[action]
    
    def _food_eaten(self):
        return self.snake.get_head_position() == self.food.position

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.starfighter = Starfighter() #player
        self.asteroids = [self._make_asteroid() for _ in range(MAX_ASTEROIDS)]

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
