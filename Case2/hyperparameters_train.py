import os

import gymnasium as gym
import numpy as np


import KiwiGym_env_CS2

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# on directory run: tensorboard --logdir=logs/hyperparameters/ --port=6006, then check on http://localhost:6006.

if __name__=="__main__":
    
    for eps_envs in [2, 3]:
        for lr in [0.0001, 0.001, 0.01]:
            for ec in [0.001, 0.01]:
                for cp in [True, False]:

                    timesteps = 90000
                    step_per_episode = 10
                    ns = step_per_episode * eps_envs
                    bs = step_per_episode * eps_envs * 8
                    env = make_vec_env("kiwiGym-CS2", n_envs=8, vec_env_cls=SubprocVecEnv)

                    model = PPO(
                        "MlpPolicy", 
                        env, 
                        learning_rate=lr, 
                        ent_coef= ec,
                        n_steps=ns, 
                        batch_size=bs,  
                        policy_kwargs=dict(net_arch=[128, 128]) if cp else None, 
                        verbose=1, 
                        tensorboard_log="./logs/hyperparameters/",
                        device="cpu"
                    )

                    # Train
                    model.learn(total_timesteps=int(timesteps), tb_log_name=f"ns_{ns}_lr_{lr}_ec_{ec}_cp_{cp}")

                    save_dir = "saved_models/hyperparameters/"
                    os.makedirs(save_dir, exist_ok=True)

                    model.save(os.path.join(save_dir, f"ns_{ns}_lr_{lr}_ec_{ec}_cp_{cp}.zip"))
