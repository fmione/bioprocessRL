import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import gymnasium as gym
import numpy as np
import keras
from keras import ops
from keras import layers
import KiwiGym_createEnv_v5
import tensorflow as tf

from stable_baselines3 import PPO
# %% Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards

env = gym.make("kiwiGym-v5")




policy_kwargs = dict(net_arch=[1024, 256,64] ) # 2048, 4096, 8192, 16384, 32768

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=1e-3, 
    ent_coef= 0.001,
    n_steps=11,  # Full episode
    batch_size=11,  # Matches episode length
    gamma=0.99, 
    gae_lambda=0.95, 
    clip_range=0.2,  
    policy_kwargs=policy_kwargs, 
    verbose=1, 
    tensorboard_log="./ppo_logs/",
    device="cpu"
)

#on directory run: tensorboard --logdir=ppo_logs --port=6006, then check on http://localhost:6006.



model.learn(total_timesteps=int(11*1e3))

save_dir = "saved_models/ppo_discrete"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, "ppo_agent"))

# %%
obs,_ = env.reset()

for _ in range(11):  # One full episode
    action, _ = model.predict(obs)  # MultiDiscrete actions
    obs, reward, done, _,_ = env.step(action)
    print(action)
    if done:
        env.render()
        break
