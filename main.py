import gymnasium as gym
import ale_py
import os
import gc
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Fix for Gymnasium 1.0+
gym.register_envs(ale_py)

def train():
    print("Setting up Training Environment...")
    # Use a simpler render mode for training to save RAM
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = AtariWrapper(env) 

    # REDUCED buffer_size to 5000 to prevent GitHub Action Memory crashes
    # INCREASED learning_starts to ensure we have enough data before training
    model = DQN("CnnPolicy", 
                env, 
                verbose=1, 
                buffer_size=5000, 
                learning_starts=2000,
                optimize_memory_usage=True, # SB3 optimization for RAM
                device="cpu")

    print("Training started (Target: 100k steps)...")
    try:
        model.learn(total_timesteps=100000, log_interval=10)
        model.save("dqn_pong_model")
    except Exception as e:
        print(f"Training interrupted: {e}")
        model.save("dqn_pong_model_partial")
    finally:
        env.close()
        gc.collect() # Force clear memory

def record():
    if not os.path.exists("dqn_pong_model.zip") and not os.path.exists("dqn_pong_model_partial.zip"):
        print("No model found to record.")
        return

    print("Recording evaluation video...")
    eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(eval_env, 
                                        video_folder="videos", 
                                        name_prefix="final_run",
                                        disable_logger=True)
    
    eval_env = AtariWrapper(eval_env)
    
    # Load whichever model we managed to save
    path = "dqn_pong_model" if os.path.exists("dqn_pong_model.zip") else "dqn_pong_model_partial"
    model = DQN.load(path)
    
    obs, _ = eval_env.reset()
    done = False
    # Limit recording to 5000 steps so the video file isn't massive
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            break
    
    eval_env.close()

if __name__ == "__main__":
    train()
    record()
