import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless pygame

import pygame
import numpy as np
import torch
import imageio
import pickle

from skrl.envs.torch import wrap_env
from skrl.models.torch import Model, GaussianMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

# -----------------
# Game Environment
# -----------------

class TagEnv:
    def __init__(self, size=500, max_steps=500):
        self.size = size
        self.max_steps = max_steps
        self.step_count = 0

        pygame.init()
        self.screen = pygame.Surface((size, size))

        self.reset()

    def reset(self):
        self.red = np.array([50.0, 50.0])
        self.blue = np.array([450.0, 450.0])
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.red / self.size, self.blue / self.size])

    def step(self, action_red, action_blue):
        speed = 4

        self.red += action_red * speed
        self.blue += action_blue * speed

        self.red = np.clip(self.red, 0, self.size)
        self.blue = np.clip(self.blue, 0, self.size)

        dist = np.linalg.norm(self.red - self.blue)

        reward_red = -dist * 0.01
        reward_blue = dist * 0.01

        if dist < 20:
            reward_red += 10
            reward_blue -= 10
            done = True
        else:
            done = False

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward_red, reward_blue, done

    def render(self):
        self.screen.fill((30, 30, 30))
        pygame.draw.rect(self.screen, (255, 50, 50), (*self.red, 20, 20))
        pygame.draw.rect(self.screen, (50, 50, 255), (*self.blue, 20, 20))
        return pygame.surfarray.array3d(self.screen)


# -----------------
# Neural Network
# -----------------

class Policy(GaussianMixin, Model):
    train()
