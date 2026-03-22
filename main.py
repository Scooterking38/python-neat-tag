import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import numpy as np
import torch
import imageio

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
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
        red_rect = pygame.Rect(int(self.red[0]), int(self.red[1]), 20, 20)
        blue_rect = pygame.Rect(int(self.blue[0]), int(self.blue[1]), 20, 20)
        pygame.draw.rect(self.screen, (255, 50, 50), red_rect)
        pygame.draw.rect(self.screen, (50, 50, 255), blue_rect)
        return pygame.surfarray.array3d(self.screen)

# -----------------
# Neural Networks
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
            torch.nn.Linear(64, self.num_actions),
            torch.nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
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
    obs_space = 4
    act_space = 2

    # RED agent
    policy_red = Policy(obs_space, act_space, device)
    value_red = Value(obs_space, act_space, device)
    memory_red = RandomMemory(memory_size=10000, num_envs=1, device=device)
    models_red = {"policy": policy_red, "value": value_red}

    # BLUE agent
    policy_blue = Policy(obs_space, act_space, device)
    value_blue = Value(obs_space, act_space, device)
    memory_blue = RandomMemory(memory_size=10000, num_envs=1, device=device)
    models_blue = {"policy": policy_blue, "value": value_blue}

    cfg = PPO_DEFAULT_CONFIG.copy()

# ensure critical numeric keys are integers/floats
    for key in ["learning_epochs", "mini_batches", "rollouts", "write_interval"]:
        if key in cfg:
            cfg[key] = int(cfg[key])

# optional: ensure other floats are floats
    for key in ["lr", "clip_range", "value_loss_coeff"]:
        if key in cfg:
            cfg[key] = float(cfg[key])
    cfg["learning_epochs"] = 4
    cfg["mini_batches"] = 2
    cfg["rollouts"] = 1024

    agent_red = PPO(models=models_red, memory=memory_red, cfg=cfg,
                    observation_space=obs_space, action_space=act_space, device=device)
    agent_blue = PPO(models=models_blue, memory=memory_blue, cfg=cfg,
                     observation_space=obs_space, action_space=act_space, device=device)

    frames = []
    timestep = 0
    total_timesteps = 100000

    for episode in range(100):
        obs = env.reset()
        done = False
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action_red = agent_red.act(state, timestep, total_timesteps)[0].detach().cpu().numpy()[0]
            action_blue = agent_blue.act(state, timestep, total_timesteps)[0].detach().cpu().numpy()[0]

            next_obs, reward_red, reward_blue, done = env.step(action_red, action_blue)
            frame = env.render()
            frames.append(frame)

            next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            done_tensor = torch.tensor([done], dtype=torch.bool).to(device)
            truncated_tensor = torch.tensor([False], dtype=torch.bool).to(device)
            infos = [{}]

            agent_red.record_transition(
                states=state,
                actions=torch.tensor(action_red, dtype=torch.float32).unsqueeze(0).to(device),
                rewards=torch.tensor([[reward_red]], dtype=torch.float32).to(device),
                next_states=next_state,
                terminated=done_tensor,
                truncated=truncated_tensor,
                infos=infos,
                timestep=timestep,
                timesteps=total_timesteps
            )

            agent_blue.record_transition(
                states=state,
                actions=torch.tensor(action_blue, dtype=torch.float32).unsqueeze(0).to(device),
                rewards=torch.tensor([[reward_blue]], dtype=torch.float32).to(device),
                next_states=next_state,
                terminated=done_tensor,
                truncated=truncated_tensor,
                infos=infos,
                timestep=timestep,
                timesteps=total_timesteps
            )

            obs = next_obs
            timestep += 1

        agent_red.update()
        agent_blue.update()
        print(f"Episode {episode} done")

    imageio.mimsave("training.mp4", frames, fps=30)
    torch.save(policy_red.state_dict(), "red_agent.pkl")
    torch.save(policy_blue.state_dict(), "blue_agent.pkl")


if __name__ == "__main__":
    train()
