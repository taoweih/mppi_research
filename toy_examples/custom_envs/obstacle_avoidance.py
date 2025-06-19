from typing import Optional

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


DEFAULT_X = 50
DEFAULT_Y = 50


class ObstacleAvoidanceEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    def __init__(self, render_mode: Optional[str] = None, env = "default"):
        self.max_speed = 8
        self.max_acceleration = 3

        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True

        self.width = 600
        self.height = 400


        self.action_space = spaces.Box(
            low=-1*self.max_acceleration, high=self.max_acceleration, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
        low=np.array([0, 0, -1*self.max_speed, -1*self.max_speed], dtype=np.float32),
        high=np.array([self.width, self.height, self.max_speed, self.max_speed], dtype=np.float32),
        dtype=np.float32)

        self.env = env

        if self.env == "default":
            self.state = np.array([50.0, 50.0, 0, 0]) #start position
            self.goal = np.array([550.0, 350.0])
            self.obstacles = [
                pygame.Rect(200, 100, 100, 300),
                pygame.Rect(400, 200, 100, 50),
            ]
        elif self.env == "U":
            self.state = np.array([330.0, 200.0, 0 ,0]) #start position
            self.goal = np.array([550.0, 200.0])
            self.obstacles = [
                pygame.Rect(200, 150, 200, 10),
                pygame.Rect(200, 250, 200, 10),
                pygame.Rect(390, 160, 10,  90),
            ]
    
        self.obs_map = np.zeros((600,400))
        for obs in self.obstacles:
            self.obs_map[obs.left:obs.left + obs.width,obs.top:obs.top + obs.height] = 1

        self.render_states = None
        self.render_policy = None

    def step(self, u):
        car_x, car_y, speed_x, speed_y = self.state
        u = np.clip(u, -1*self.max_acceleration, self.max_acceleration)
        speed_x = np.clip(speed_x + u[0], -1*self.max_speed, self.max_speed)
        speed_y = np.clip(speed_y + u[1], -1*self.max_speed, self.max_speed)

        new_x = car_x + speed_x
        new_y = car_y + speed_y

        new_x = int(new_x)
        new_y = int(new_y)
        if new_x < 0 or new_x > 599 or new_y < 0 or new_y > 399 or self.obs_map[new_x,new_y] == 1 :
            new_x = car_x
            new_y = car_y

        self.state = np.array([new_x, new_y, speed_x,speed_y])

        if self.render_mode == "human":
            self.render()

        costs = 0

        return self._get_obs(), -costs, False, False, {}
    
    def reset(self, seed=None, options=None):
        if self.env == "default":
            self.state = np.array([50.0, 50.0, 0, 0]) #start position

        elif self.env == "U":
            self.state = np.array([330.0, 200.0, 0, 0]) #start position

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}


    def _get_obs(self):
        x, y, dx, dy= self.state
        return np.array([x, y, dx, dy], dtype=np.float32)
    
    def set_render_info(self, states, policy):
        self.render_states = states[:,:,0:2].astype("int")
        self.render_policy = policy[:,0:2].astype("int")

    def render(self):
        freq = 1000
        counter = 0

        states= self.render_states

        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Obstacle Avoidance Env")
                self.screen = pygame.display.set_mode(
                    (self.width, self.height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()


        background = pygame.Surface((self.width, self.height))
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        background.fill((255, 255, 255))  #white
        for obs in self.obstacles:
            pygame.draw.rect(background, (0, 0, 0), obs)
        pygame.draw.circle(background, (255, 0, 0), self.goal, 8) #goal

        overlay.fill((0, 0, 0, 0))

        pygame.draw.circle(overlay, (0, 0, 255), self.state[0:2], 8)
        if states is not None:
            for traj in states:
                if counter % freq == 0:
                    for i in range(len(traj)):
                        if i > 0:
                            pygame.draw.line(overlay, (0,0,0), traj[i-1], traj[i], 1)
                counter +=1
        
        if self.render_policy is not None:
            for i in range(len(self.render_policy)):
                if i > 0:
                    pygame.draw.line(overlay, (255,0,255), self.render_policy[i-1], self.render_policy[i], 1)
       
        self.screen.blit(background, (0, 0))
        self.screen.blit(overlay, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

