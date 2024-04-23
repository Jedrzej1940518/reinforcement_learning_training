
from dataclasses import dataclass
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
FIGHTER_MAX_ACCELERATION = 1
FIGHTER_COOLDOWN  = 10
FIGHTER_MAX_HP = 10
FIGHTER_MAX_ROTATION_SPEED = 0.1 #radians

ASTEROID_MAX_SIZE = 20
ASTEROID_MAX_VELOCITY = 8
ASTEROID_MAX_HP = 3
MAX_ASTEROIDS = 4

class Vector:
    
    def __init__(self, magnitude=0, direction=0):
        self.magnitude = magnitude
        self.direction = direction

    def cartesian(self):
        r = self.magnitude
        return r* math.cos(self.direction), r*math.sin(self.direction)

    def from_cartesian(self, x, y):
        self.magnitude = math.sqrt(x**2 + y**2)
        self.direction = math.atan2(y, x)
        return self

    def __iter__(self):
        return iter((self.magnitude, self.direction))

@dataclass
class Point:
    x: float
    y: float      #angle in radians

class RotationComponent:
    def __init__(self, max_rotation_speed, rotation):
        self.max_rotation_speed = max_rotation_speed
        self.rotation = rotation
        self.rotation_speed = 0

    def rotate(self, rotation_speed):    
        self.rotation_speed = min(rotation_speed, self.max_rotation_speed) if rotation_speed > 0 else max(rotation_speed, -self.max_rotation_speed)

    def update(self):
        self.rotation += self.rotation_speed
        
        if self.rotation < 0:
            self.rotation += 2* math.pi
        else:
            self.rotation -= 2* math.pi

class MovementComponent:
    
    def __init__(self, position:Point, velocity: Vector, max_speed, max_acceleration = 0):
        
        self.position = position
        self.velocity = velocity
        self.acceleration = Vector()

        self.max_speed = max_speed
        self.max_acceleration = max_acceleration

    def accelerate(self, acceleration:Vector):
        self.acceleration = acceleration
        self.acceleration.magnitude = min(self.acceleration.magnitude, self.max_acceleration)

    def _update_velocity(self):
        if self.max_acceleration == 0:
            return 

        x, y = self.velocity.cartesian()
        xa, ya = self.acceleration.cartesian()
        
        self.velocity = Vector().from_cartesian(x+xa, y+ya)
        self.velocity.magnitude = min(self.velocity.magnitude, self.max_speed)
        print(f"debug x,y: {x,y}, xa, ya{xa, ya}, new velocity {self.velocity.cartesian()}")

    def update(self):
        x,y = self.velocity.cartesian()
        self.position.x += x
        self.position.y += y
        self._update_velocity()

    def get_normalized_obs(self):
        return [self.position.x / WIDTH, self.position.y / HEIGHT, self.velocity.magnitude / self.max_speed, self.velocity.direction / math.pi]


class Asteroid:
    def __init__(self, movement_component : MovementComponent, size: int, hp: int):
        self.movement_component = movement_component
        self.size = size
        self.max_hp = hp
        self.hp_left = hp

    def update(self):
        self.movement_component.update()

    def get_normalized_obs(self):
        return self.movement_component.get_normalized_obs() + [self.size / ASTEROID_MAX_SIZE, self.hp_left / ASTEROID_MAX_HP]

class Starfighter:
    def __init__(self, movement_component: MovementComponent, rotation_component: RotationComponent):
        self.movement_component = movement_component
        self.rotation_component = rotation_component
        self.hp = FIGHTER_MAX_HP
        self.cooldown = 0
        self.shooting = False

    def instruct(self, acceleration, rotation_speed, shooting):
        self.movement_component.accelerate(acceleration)
        self.rotation_component.rotate(rotation_speed)
        self.shooting = shooting

    def update(self):
        self.movement_component.update()
        self.rotation_component.update()
        
        pass

    def get_normalized_obs(self):
        """
        fighter x, fighter y, fighter speed, fighter speed direction, fighter rotation, fighter cooldown, fighter hp left
        """
        return self.movement_component.get_normalized_obs() + [self.rotation_component.rotation / math.pi, self.cooldown / FIGHTER_COOLDOWN, self.hp / FIGHTER_MAX_HP]


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
        -size -> 0....ASTEROID_MAX_SIZE
        -velocity magnitude -> 0.....ASTEROID_MAX_VELOCITY
        -velocity direction -> -3.14....3.14
        -hp left (numbers of shots to kill) -> 0.....ASTEROID_MAX_HP
        """

        asteroid_low = [0,0, 0, -3.14, 0, 0]
        asteroid_high = [WIDTH, HEIGHT, ASTEROID_MAX_VELOCITY, math.pi, ASTEROID_MAX_SIZE, ASTEROID_MAX_HP]

        high = fighter_high + asteroid_high * MAX_ASTEROIDS 
        high = np.array(high).astype(np.float32)
        low = fighter_low + asteroid_low * MAX_ASTEROIDS
        low = np.array(low).astype(np.float32)
        """
        obs space:
        fighter x, fighter y, fighter speed, fighter speed direction, fighter rotation, fighter cooldown, fighter hp left, + asteroid_num* [asteroid x, asteroid y, asteroid speed, asteroid speed dir, asteroid size, asteroid hp left]
        """
        self.observation_space = spaces.Box(low, high, shape=(len(high),), dtype=float) 
        
        """
        action space:
        accelerate -> 0...1 -> acceleration force (percentage)
        acceleration dir -> -3.14...3.14 -> direction of acceleration (radians)
        rotate -> -1.....1 -> left, right (percentages)
        shoot -> -1...1 -> <0 no shoot, >0 shoot
        """
        action_space_low = np.array([0,-math.pi, -1, -1]).astype(np.float32)
        action_space_high = np.array([1,math.pi, 1, 1]).astype(np.float32)
        self.action_space = spaces.Box(action_space_low, action_space_high, shape=(len(action_space_high),), dtype=float) 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    
    def _random_point_on_square(self) -> Point:
        edge = random.randint(1, 4)
        t = random.uniform(0, EDGE_L)
        
        if edge == 1:   #top
            return Point(t, 0)
        elif edge == 2: #right
            return Point(EDGE_L, t) 
        elif edge == 3: #bottom
            return Point(t, EDGE_L)
        elif edge == 4: #left
            return Point(0, t)
        
        return Point(0,0)
    
    def _angle_between_points(self, a: Point, b: Point):
        # Direction vector
        dx = b.x - a.x
        dy = b.y - a.y
        return math.atan2(dy, dx) 


    def _make_asteroid(self) -> Asteroid:
        
        pos = self._random_point_on_square()
        angle_to_starship = self._angle_between_points(pos, self.starfighter.movement_component.position)
        velocity = Vector(random.uniform(0.1, ASTEROID_MAX_VELOCITY), random.uniform(-math.pi/6, math.pi/6) + angle_to_starship) #velocity in general direction of starship +- 30 degrees
        asteroid_movement_component = MovementComponent(pos, velocity, ASTEROID_MAX_VELOCITY)
        
        size = random.randint(1, ASTEROID_MAX_SIZE)
        hp = random.randint(1, ASTEROID_MAX_HP)             #todo probably scale with size
        return Asteroid(asteroid_movement_component, size, hp)

    def _get_obs(self):

        obs = self.starfighter.get_normalized_obs()
        for asteroid in self.asteroids:
            obs += asteroid.get_normalized_obs() 

        print("normalized_obs: ", obs)

        return np.array(obs)

    def _get_info(self):
        return {"Info" : 0}
    
    def _instruct_starfighter(self, action):
        """
        action space:
        accelerate -> 0...1 -> acceleration force (percentage)
        acceleration dir -> -3.14...3.14 -> direction of acceleration (radians)
        rotate -> -1.....1 -> left, right (percentages)
        shoot -> -1...1 -> <0 no shoot, >0 shoot
        """

        acceleration = Vector(action[0] * FIGHTER_MAX_ACCELERATION, action[1])
        rotation_speed = action[2] * FIGHTER_MAX_ROTATION_SPEED
        shooting = action[3] > 0
        self.starfighter.instruct(acceleration, rotation_speed, shooting)


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
       
        self._instruct_starfighter(action)
        self.starfighter.update()
        for a in self.asteroids:
            a.update()


        terminated = self.starfighter.hp <= 0
        reward = 1
     
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
