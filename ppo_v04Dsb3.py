import os

import gymnasium as gym
import numpy as np

import KiwiGym_createEnv_v4D

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


# %% Configuration parameters for the whole setup
#on directory run: tensorboard --logdir=ppo_logs --port=6006, then check on http://localhost:6006.

env = gym.make("kiwiGym-v4D")
lr = 5e-4
ns = 110
bs = 55
ec = 0.01
custom_policy = False
policy_kwargs = dict(net_arch=[512, 256]) 

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=lr, 
    ent_coef= ec, # por defecto 0.01. Está la opcion "auto" para que varie. Para explorar.
    n_steps=ns, 
    batch_size=bs,  
    # gamma=0.99, # por defecto 0.99 (peso a las recompenzas a largo plazo o inmediatas)
    # gae_lambda=0.95, # por defecto 0.95 
    # clip_range=0.2,  # por defecto 0.2
    policy_kwargs=policy_kwargs if custom_policy else None, 
    verbose=1, 
    tensorboard_log="./ppo_env4D/",
    device="cpu"  ,
    #New hyperparam:
    # n_epochs=20, # por defecto 10
    # target_kl=0.05 # es un criterio de parada si la politica cambia bastante. Por defecto no se usa
)
# %% Train
model.learn(total_timesteps=int(800000), tb_log_name=f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}")

save_dir = "saved_models/ppo_env4D"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}"))


# %% Best model evaluation

model_name = f"lr_{lr}_ns_{ns}_bs_{bs}_cp_{custom_policy}_ec_{ec}"

model = PPO.load(f"saved_models/ppo_env4D/{model_name}", print_system_info=True, env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True)

print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")

# Resultado obtenido para 1000 eps - Recompensa media: 3.15 +/- 0.42