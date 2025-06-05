import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import KiwiGym_createEnv_v4F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train_model(n_envs, lr, ec, cp):

    env = make_vec_env("kiwiGym-v4F", n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    timesteps = 108000
    step_per_episode = 9
    ns = step_per_episode
    bs = step_per_episode * n_envs

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=lr, 
        ent_coef= ec, 
        n_steps=ns, 
        batch_size=bs,  
        policy_kwargs=dict(net_arch=[512, 256])  if cp else None, 
        verbose=1, 
        tensorboard_log="./ppo_logs/",
        device="cpu",
    )

    # Train
    model.learn(total_timesteps=int(timesteps), tb_log_name=f"n_envs_{n_envs}_lr_{lr}_ec_{ec}_cp_{cp}")

    print("Training completed. Saving model...")

    save_dir = "saved_models/ppo_env4F"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"n_envs_{n_envs}_lr_{lr}_ec_{ec}_cp_{cp}.zip"))

    print(f"Model saved")


if __name__ == '__main__':    

    n_envs = int(sys.argv[1])
    lr = float(sys.argv[2])
    ec = float(sys.argv[3])
    cp = bool(sys.argv[4])

    print(f"Training with n_envs={n_envs}, lr={lr}, ec={ec}, cp={cp}")
    print("Starting training...")
    train_model(n_envs, lr, ec, cp)
