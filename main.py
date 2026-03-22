import gymnasium as gym
import ale_py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# 1. Register Atari Envs (Required for Gym 1.0+)
gym.register_envs(ale_py)

def train():
    # Setup training environment with standard Atari preprocessing
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = AtariWrapper(env) 

    # 2. Define the Model
    # 'CnnPolicy' automatically handles the image-to-layers logic
    model = DQN("CnnPolicy", env, 
                verbose=1, 
                buffer_size=10000, 
                learning_starts=1000,
                device="cpu") # Actions runners usually don't have GPUs

    print("Training started...")
    model.learn(total_timesteps=100000)
    model.save("dqn_pong_model")
    env.close()

def record():
    print("Recording evaluation video...")
    # Create evaluation env with Video Recorder
    eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(eval_env, 
                                        video_folder="videos", 
                                        name_prefix="final_run")
    
    # We must apply the same AtariWrapper so the input shapes match
    eval_env = AtariWrapper(eval_env)
    
    model = DQN.load("dqn_pong_model")
    
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    
    eval_env.close()
    print("Video saved to ./videos/")

if __name__ == "__main__":
    train()
    record()
