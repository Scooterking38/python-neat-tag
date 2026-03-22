import gymnasium as gym
import ale_py
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# 1. Register Atari Envs (Crucial for Gym 1.0+)
gym.register_envs(ale_py)

def train():
    print("Initializing Training Environment...")
    # Use render_mode=None for training to speed up and avoid buffer issues
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = AtariWrapper(env) 

    model = DQN("CnnPolicy", 
                env, 
                verbose=1, 
                buffer_size=5000, 
                learning_starts=500, # Lowered for faster testing
                device="cpu")

    print("Training...")
    model.learn(total_timesteps=20000) # Lowered to 20k to ensure it finishes
    model.save("dqn_pong_model")
    env.close()

def record():
    print("Initializing Recording Environment...")
    # RecordVideo requires rgb_array
    eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    
    # Wrap with RecordVideo BEFORE AtariWrapper to capture high-res color
    eval_env = gym.wrappers.RecordVideo(eval_env, 
                                        video_folder="videos", 
                                        name_prefix="final_run")
    
    eval_env = AtariWrapper(eval_env)
    
    if os.path.exists("dqn_pong_model.zip"):
        model = DQN.load("dqn_pong_model")
        obs, _ = eval_env.reset()
        for _ in range(1000): # Record 1000 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                break
    eval_env.close()

if __name__ == "__main__":
    train()
    record()
