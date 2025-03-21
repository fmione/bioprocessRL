import os

import gymnasium as gym
import numpy as np

import KiwiGym_createEnv_v4

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# %% Configuration parameters for the whole setup
#on directory run: tensorboard --logdir=ppo_logs --port=6006, then check on http://localhost:6006.

# Create multiple environments
n_envs = 25  # Set the number of parallel environments
if n_envs>1:   
    env = make_vec_env("kiwiGym-v4", n_envs=n_envs)
else:
    env = gym.make("kiwiGym-v4")
#
policy_kwargs = dict(net_arch=[256,128,64] ) # 

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=1e-3, 
    ent_coef= 0.001,
    n_steps=11,  # Full episode
    batch_size=int(11*n_envs),  
    gamma=0.99, 
    gae_lambda=0.95,
    clip_range=0.2,  
    # policy_kwargs=policy_kwargs, 
    verbose=1, 
    tensorboard_log="./ppo_logs/",
    device="cpu"  ,
    #New hyperparam:
    n_epochs=10, 
    # target_kl=0.05
)
# %% Train
model.learn(total_timesteps=int(11*5e2*n_envs))
    
save_dir = "saved_models/ppo_discrete"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, "ppo_agent6"))
# %% Test
env = gym.make("kiwiGym-v4")
obs,_ = env.reset()
TH_store=env.unwrapped.kiwiGym.TH_param

for i in range(11):  # One full episode
    # obs[1:]=(1+0.1*np.random.random(obs[1:].shape))*obs[1:]
    action, _ = model.predict(obs,deterministic=True)  
    obs, reward, done, _,_ = env.step(action)
    print(action)    
    if done:
        env.render()
        break

# obs,_ = env.reset(options=TH_store)


# for i in range(11):  # One full episode
#     # obs[1:]=(1+0.1*np.random.random(obs[1:].shape))*obs[1:]
#     action =np.array([0])# model.predict(obs,deterministic=True)  
#     obs, reward, done, _,_ = env.step(action)
#     print(action)    
#     if done:
#         env.render()
#         break