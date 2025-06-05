import os

import gymnasium as gym
import numpy as np

import KiwiGym_createEnv_v4F

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


# %% Configuration parameters for the whole setup
#on directory run: tensorboard --logdir=ppo_logs --port=6006, then check on http://localhost:6006.

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__=="__main__":
    
    for n_envs in [2, 3, 4]:
        for lr in [0.0001, 0.001, 0.01]:
            for ec in [0.001, 0.01]:
                for cp in [True, False]:

                    timesteps = 90000
                    step_per_episode = 10
                    ns = step_per_episode * n_envs
                    bs = step_per_episode * n_envs * 8
                    env = make_vec_env("kiwiGym-v4F", n_envs=8, vec_env_cls=SubprocVecEnv)

                    model = PPO(
                        "MlpPolicy", 
                        env, 
                        learning_rate=lr, 
                        ent_coef= ec,
                        n_steps=ns, 
                        batch_size=bs,  
                        policy_kwargs=dict(net_arch=[128, 128]) if cp else None, 
                        verbose=1, 
                        tensorboard_log="./logs/ppo_oneMBR/2nd/",
                        device="cpu"
                    )
                    # %% Train
                    model.learn(total_timesteps=int(timesteps), tb_log_name=f"ns_{ns}_lr_{lr}_ec_{ec}_cp_{cp}")

                    save_dir = "saved_models/oneMBR4F/2nd/"
                    os.makedirs(save_dir, exist_ok=True)

                    model.save(os.path.join(save_dir, f"ns_{ns}_lr_{lr}_ec_{ec}_cp_{cp}.zip"))


# %% Best model evaluation

# model_name = f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}"

# model = PPO.load(f"saved_models/ppo_env4F/{model_name}", print_system_info=True, env=env)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

# print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")




# %% Test model
# env = gym.make("kiwiGym-v4F")
# model_name = "lr_0.0005_ns_110_bs_55_cp_False_ec_0.01"
# model = PPO.load(f"saved_models/ppo_env4E/{model_name}", print_system_info=True, env=env)

# for it in range(10):
#     obs,_ = env.reset()
#     for i in range(11):  # One full episode
#         action, _ = model.predict(obs, deterministic=True)  
#         obs, reward, done, _,_ = env.step(action)
#         print(action)    
#         if done:
#             env.render()
#             break
# %%
