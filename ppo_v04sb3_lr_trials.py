import os

import gymnasium as gym
import numpy as np

import KiwiGym_createEnv_v4
# import KiwiGym_createEnv_v4C

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


# %% Configuration parameters for the whole setup
#on directory run: tensorboard --logdir=ppo_logs --port=6006, then check on http://localhost:6006.

# Create multiple environments
# n_envs = 1  # Set the number of parallel environments
# if n_envs>1:   
#     env = make_vec_env("kiwiGym-v4", n_envs=n_envs)
# else:
#     env = gym.make("kiwiGym-v4")


# policy_kwargs = dict(net_arch=[512, 256]) 


# # for lr in [5e-4, 1e-3, 1e-4, 1e-2]:
# #     for ns in [110, 88, 55]:	
# #         for bs in [88, 55, 33]:
# #             for custom_policy in [True, False]:
# #                 if ns == 55 and bs == 88:
# #                     continue
# for lr in [1e-3, 1e-4, 1e-2]:
#     for ns in [88]:	
#         for bs in [55]:
#             for custom_policy in [False, True]:
#                 for ec in [0.01, 0.001]:    
                        
#                     if not custom_policy and lr == 1e-3:
#                         continue
#                     else:

#                         model = PPO(
#                             "MlpPolicy", 
#                             env, 
#                             learning_rate=lr, 
#                             ent_coef= ec, # por defecto 0.01. Está la opcion "auto" para que varie. Para explorar.
#                             n_steps=ns, 
#                             batch_size=bs,  
#                             # gamma=0.99, # por defecto 0.99 (peso a las recompenzas a largo plazo o inmediatas)
#                             # gae_lambda=0.95, # por defecto 0.95 
#                             # clip_range=0.2,  # por defecto 0.2
#                             policy_kwargs=policy_kwargs if custom_policy else None, 
#                             verbose=1, 
#                             tensorboard_log="./ppo_env4/",
#                             device="cpu"  ,
#                             #New hyperparam:
#                             # n_epochs=20, # por defecto 10
#                             # target_kl=0.05 # es un criterio de parada si la politica cambia bastante. Por defecto no se usa
#                         )
#                         # %% Train
#                         model.learn(total_timesteps=int(80000), tb_log_name=f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}")

#                         save_dir = "saved_models/ppo_env4"
#                         os.makedirs(save_dir, exist_ok=True)

#                         model.save(os.path.join(save_dir, f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}"))


# %% Test
# model_name = "lr_0.0001_ns_88_bs_55_cp_False_ec_0.01"
# model_name = "lr_0.0005_ns_110_bs_55_cp_False"
model_name = "lr_0.0005_ns_110_bs_55_cp_True"
env = gym.make("kiwiGym-v4")
# for lr in [5e-4, 1e-4]:

model = PPO.load(f"saved_models/ppo_env4/{model_name}", print_system_info=True, env=env)

#  Train
lr = 1e-4
model.learning_rate = lr
model.learn(total_timesteps=int(800000), tb_log_name=f"2nd_lr_{lr}_{model_name}")

save_dir = "saved_models/ppo_env4_2nd_training"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, f"2nd_lr_{lr}_{model_name}"))


# # train again with lr = 5e-5
# model_name = "lr_0.0001_ns_88_bs_55_cp_False_ec_0.01"

# model = PPO.load(f"saved_models/ppo_env4/{model_name}", print_system_info=True, env=env)

# #  Train
# lr = 5e-5
# model.learning_rate = lr
# model.learn(total_timesteps=int(800000), tb_log_name=f"2nd_lr_{lr}_{model_name}")

# save_dir = "saved_models/ppo_env4_2nd_training"
# os.makedirs(save_dir, exist_ok=True)

# model.save(os.path.join(save_dir, f"2nd_lr_{lr}_{model_name}"))

#%%

# # %% Test
model = PPO.load("saved_models/ppo_env4_2nd_training/2nd_lr_0.0001_lr_0.0005_ns_110_bs_55_cp_False", print_system_info=True)

env = gym.make("kiwiGym-v4")

for it in range(10):
    obs,_ = env.reset()
    #[20,20,20,20,20,20,20,20,20,14,12]
    for i in range(11):  # One full episode
        # obs[:11] = np.tile([1], 11)
        # obs = np.concatenate((np.tile([1], 11), obs[11:] * (0*np.random.randn(len(obs[11:])))))
        # obs[11:] = obs[11:] * np.random.randn(len(obs[11:]))
        # obs = np.tile([1], len(obs))
        # obs[11:] = np.tile([1], len(obs[11:]))
        # print(obs[:34])
        action, _ = model.predict(obs, deterministic=True)  
        obs, reward, done, _,_ = env.step(action)
        print(action)    
        if done:
            env.render()
            break




# %% Best model evaluation

# model_name = "2nd_lr_0.0001_lr_0.0005_ns_110_bs_55_cp_False"
# env = gym.make("kiwiGym-v4")


# model = PPO.load(f"saved_models/ppo_env4_2nd_training/{model_name}", print_system_info=True, env=env)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True)

# print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")

# Resultado obtenido para 1000 eps - Recompensa media: 3.13 +/- 0.68
