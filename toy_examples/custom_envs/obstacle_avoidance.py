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

        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True

        self.width = 600
        self.height = 400


        self.action_space = spaces.Box(
            low=-0, high=600, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
        low=np.array([0, 0], dtype=np.float32),
        high=np.array([self.width, self.height], dtype=np.float32),
        dtype=np.float32)

        self.env = env

        if self.env == "default":
            self.state = np.array([50.0, 50.0]) #start position
            self.goal = np.array([550.0, 350.0])
            self.obstacles = [
                pygame.Rect(200, 100, 100, 300),
                pygame.Rect(400, 200, 100, 50),
            ]
        elif self.env == "U":
            self.state = np.array([330.0, 200.0]) #start position
            self.goal = np.array([500.0, 200.0])
            self.obstacles = [
                pygame.Rect(300, 150, 100, 10),
                pygame.Rect(300, 250, 100, 10),
                pygame.Rect(390, 160, 10,  90),
            ]
    
        self.obs_map = np.zeros((600,400))
        for obs in self.obstacles:
            self.obs_map[obs.left:obs.left + obs.width,obs.top:obs.top + obs.height] = 1

    def step(self, u):
        car_x, car_y = self.state

        u = np.clip(u, -10, 10)
        new_x = car_x + u[0]
        new_y = car_y + u[1]
        new_x = int(new_x)
        new_y = int(new_y)
        if new_x < 0 or new_x > 599 or new_y < 0 or new_y > 399 or self.obs_map[new_x,new_y] == 1 :
            new_x = car_x
            new_y = car_y

        self.state = np.array([new_x, new_y])

        if self.render_mode == "human":
            self.render()

        costs = 0

        return self._get_obs(), -costs, False, False, {}
    
    def reset(self, seed=None, options=None):
        if self.env == "default":
            self.state = np.array([50.0, 50.0]) #start position

        elif self.env == "U":
            self.state = np.array([330.0, 200.0]) #start position

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}


    def _get_obs(self):
        x, y = self.state
        return np.array([x, y], dtype=np.float32)

    def render(self):
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

        pygame.draw.circle(overlay, (0, 0, 255), self.state, 8)
       
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

