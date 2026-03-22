import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless pygame

import pygame
import numpy as np
import torch
import imageio
import pickle

from skrl.models.torch import Model, GaussianMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG


# -----------------
# Game Environment
# -----------------
class TagEnv:
    def __init__(self, size=500, max_steps=300):
        self.size = size
        self.max_steps = max_steps
        self.step_count = 0

        pygame.init()
        self.screen = pygame.Surface((size, size))

        self.reset()

    def reset(self):
        self.red = np.array([50.0, 50.0], dtype=np.float32)
        self.blue = np.array([450.0, 450.0], dtype=np.float32)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.red / self.size, self.blue / self.size]).astype(np.float32)

    def step(self, action_red, action_blue):
        speed = 4

        self.red += action_red * speed
        self.blue += action_blue * speed

        self.red = np.clip(self.red, 0, self.size)
        self.blue = np.clip(self.blue, 0, self.size)

        dist = np.linalg.norm(self.red - self.blue)

        reward_red = -dist * 0.01
        reward_blue = dist * 0.01

        done = False
        if dist < 20:
            reward_red += 10
            reward_blue -= 10
            done = True

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
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_actions)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# -----------------
# Training
# -----------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = TagEnv()

    observation_space = 4
    action_space = 2

    policy_red = Policy(observation_space, action_space, device)
    policy_blue = Policy(observation_space, action_space, device)

    memory_red = RandomMemory(memory_size=10000, num_envs=1, device=device)
    memory_blue = RandomMemory(memory_size=10000, num_envs=1, device=device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["learning_epochs"] = 4
    cfg["mini_batches"] = 2

    agent_red = PPO(models=policy_red, memory=memory_red, cfg=cfg,
                    observation_space=observation_space, action_space=action_space, device=device)

    agent_blue = PPO(models=policy_blue, memory=memory_blue, cfg=cfg,
                     observation_space=observation_space, action_space=action_space, device=device)

    frames = []

    for episode in range(100):
        obs = env.reset()
        done = False

        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            action_red = agent_red.act(state)[0].cpu().numpy()[0]
            action_blue = agent_blue.act(state)[0].cpu().numpy()[0]

            next_obs, reward_red, reward_blue, done = env.step(action_red, action_blue)

            frame = env.render()
            frames.append(frame)

            agent_red.record_transition(state, action_red, reward_red, next_obs, done)
            agent_blue.record_transition(state, action_blue, reward_blue, next_obs, done)

            obs = next_obs

        agent_red.update()
        agent_blue.update()

        print(f"Episode {episode} done")

    imageio.mimsave("training.mp4", frames, fps=30)

    with open("red_agent.pkl", "wb") as f:
        pickle.dump(policy_red.state_dict(), f)

    with open("blue_agent.pkl", "wb") as f:
        pickle.dump(policy_blue.state_dict(), f)


if __name__ == "__main__":
    train()
