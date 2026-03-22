import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import os

# New Gymnasium v1.0+ requirement: Explicitly register Atari environments
gym.register_envs(ale_py)

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Seed for reproducibility
set_seed(42)

# 1. Define the CNN Model for Atari (Nature CNN architecture)
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Input shape expected: (4, 84, 84) due to FrameStack
        self.net = nn.Sequential(
            nn.Conv2d(self.num_observations[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def compute(self, inputs, role):
        # Normalize pixel values from [0, 255] to [0, 1]
        return self.net(inputs["states"] / 255.0), {}

def main():
    # 2. Setup Training Environment
    # Use "rgb_array" even in training to avoid display errors in headless CI
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)
    env = wrap_env(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Setup Memory and Agent
    memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device)

    models = {
        "q_network": QNetwork(env.observation_space, env.action_space, device),
        "target_q_network": QNetwork(env.observation_space, env.action_space, device)
    }

    # Initialize weights
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # Configure DQN
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 1000
    cfg["exploration"]["initial_epsilon"] = 1.0
    cfg["exploration"]["final_epsilon"] = 0.05
    cfg["exploration"]["timesteps"] = 30000
    cfg["experiment"]["directory"] = "runs/torch/ALE_Pong"

    agent = DQN(models=models, memory=memory, cfg=cfg,
                observation_space=env.observation_space, 
                action_space=env.action_space, device=device)

    # 4. Train the Agent
    print("Starting Training (Headless)...")
    cfg_trainer = {"timesteps": 100000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()
    env.close()

    # 5. Record the Results
    print("Recording evaluation video...")
    eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    
    # RecordVideo should wrap the base env to capture full resolution
    eval_env = RecordVideo(eval_env, video_folder="videos", name_prefix="pong_result", 
                           episode_trigger=lambda x: True)
    
    # Apply processing so the agent understands the input
    eval_env = AtariPreprocessing(eval_env, screen_size=84, grayscale_obs=True, frame_skip=4)
    eval_env = FrameStackObservation(eval_env, stack_size=4)
    eval_env = wrap_env(eval_env)

    agent.set_mode("eval")
    obs, _ = eval_env.reset()
    
    for _ in range(2000): # Run for max 2000 steps or until done
        action, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated.any() or truncated.any():
            break

    eval_env.close()
    print(f"Finished. Video saved in {os.path.abspath('videos')}")

if __name__ == "__main__":
    main()
