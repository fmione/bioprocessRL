import os

import gymnasium as gym
import numpy as np

import KiwiGym_createEnv_v4O

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# %% Configuration parameters for the whole setup
#on directory run then check on http://localhost:6006:
# tensorboard --logdir=ppo_logs --port=6006
    
# Create multiple environments
if __name__=="__main__":
    n_envs = 8  # Set the number of parallel environments
    n_step= 10 *3
    env = make_vec_env("kiwiGym-v4O", n_envs=n_envs,vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(net_arch=[512,264,128] ) #
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        ent_coef= 0.01,
        n_steps=n_step,  # Full episode, 4O=10
        batch_size=int(n_step*n_envs),  
        gamma=0.99, 
        gae_lambda=0.95,
        clip_range=0.2,  
        # policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log="./ppo_logs/",
        device="cpu"  ,
        #New hyperparam:
        n_epochs=10, 
        target_kl=0.05
    )
    # %% Train
    save_dir = "saved_models/ppo_agent_4O"
    os.makedirs(save_dir, exist_ok=True)
    
    # load_dir = "saved_models/ppo_agent_4O"
    # model.set_parameters(os.path.join(load_dir, "ppo_agent_4O_intermediate"),exact_match=True, device="cpu")
    

    cnt=0
    while cnt < 10 :
        try:
            model.set_env(env)
            model.learn(total_timesteps=int(10*n_step*n_envs), reset_num_timesteps=True)
            model.save(os.path.join(save_dir, "ppo_agent_4O_intermediate"))
            model=PPO.load(os.path.join(save_dir, "ppo_agent_4O_intermediate"),device="cpu")
            cnt=cnt+1
        except:
         cnt=cnt+0     
    
    model.save(os.path.join(save_dir, "ppo_agent_4O"))
    # %% Test
    env = gym.make("kiwiGym-v4O")
    obs,_ = env.reset()
    
    for i in range(10):  # One full episode
        # obs[1:]=0.0*obs[1:]
        action, _ = model.predict(obs,deterministic=True)  
        obs, reward, done, _,_ = env.step(action)
        print(action)    
        if done:
            env.render()
            break