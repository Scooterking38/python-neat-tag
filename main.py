import gymnasium as gym
import torch
import torch.nn as nn

from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordVideo
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Seed for reproducibility
set_seed(42)

# 1. Define the CNN Model for Atari
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

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
        # Normalize pixel values to [0, 1]
        return self.net(inputs["states"] / 255.0), {}

def main():
    # 2. Setup Training Environment
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
    env = FrameStack(env, 4)
    env = wrap_env(env)

    device = env.device

    # 3. Setup Memory and Agent
    memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device, replacement=False)

    models = {
        "q_network": QNetwork(env.observation_space, env.action_space, device),
        "target_q_network": QNetwork(env.observation_space, env.action_space, device)
    }

    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 10000
    cfg["exploration"]["initial_epsilon"] = 1.0
    cfg["exploration"]["final_epsilon"] = 0.05
    cfg["exploration"]["timesteps"] = 50000
    cfg["experiment"]["write_interval"] = 5000
    cfg["experiment"]["checkpoint_interval"] = 10000
    cfg["experiment"]["directory"] = "runs/torch/ALE_Pong"

    agent = DQN(models=models, memory=memory, cfg=cfg,
                observation_space=env.observation_space, action_space=env.action_space, device=device)

    # 4. Train the Agent
    print("Starting Training...")
    cfg_trainer = {"timesteps": 100000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()

    env.close()

    # 5. Record the Best Run
    print("Training Complete. Recording Video...")
    
    # Create a fresh environment for recording high-res RGB frames
    eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    
    # Wrap with RecordVideo FIRST so we capture the original colors and resolution
    eval_env = RecordVideo(eval_env, video_folder="videos", name_prefix="pong-agent", episode_trigger=lambda x: True)
    
    # Apply the same preprocessing so the agent can understand the observation
    eval_env = AtariPreprocessing(eval_env, screen_size=84, grayscale_obs=True, frame_skip=4)
    eval_env = FrameStack(eval_env, 4)
    eval_env = wrap_env(eval_env)

    agent.set_mode("eval")
    obs, _ = eval_env.reset()
    done = False
    
    # Run one full episode
    while not done:
        action, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        # Check if any environment is done
        done = terminated.any() or truncated.any()

    eval_env.close()
    print("Video recording complete. Saved to 'videos/' directory.")

if __name__ == "__main__":
    main()
