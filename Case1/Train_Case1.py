import os

import gymnasium as gym
import numpy as np

import KiwiGym_env_CS1_0
import KiwiGym_env_CS1

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# on case1 directory run: tensorboard --logdir=logs/final --port=6006, then check on http://localhost:6006.

if __name__=="__main__":
    
    # Hyperparameters
    eps_envs = 2
    lr = 0.001
    ec = 0.001
    cp = True

    total_timesteps = 100000
    step_per_episode = 10
    ns = step_per_episode * eps_envs
    bs = step_per_episode * eps_envs * 8

    for model_name in ["kiwiGym-CS1", "kiwiGym-CS1_0"]:

        # Create the environments (8 parallel environments)
        env = make_vec_env(model_name, n_envs=8, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=lr, 
            ent_coef= ec,
            n_steps=ns, 
            batch_size=bs,  
            policy_kwargs=dict(net_arch=[128, 128]) if cp else None, 
            verbose=1, 
            tensorboard_log="./logs/final2/",
            device="cpu"
        )

        # Train: checkpointing every 20000 steps
        checkpointint_steps = 5
        timesteps = total_timesteps / checkpointint_steps

        for i in range(checkpointint_steps):
            model.learn(total_timesteps=int(timesteps), tb_log_name=f"model_{model_name}", reset_num_timesteps=False)

            save_dir = f"saved_models/model_{model_name}"
            os.makedirs(save_dir, exist_ok=True)

            model.save(os.path.join(save_dir, f"{i + 1}_model_final.zip"))

