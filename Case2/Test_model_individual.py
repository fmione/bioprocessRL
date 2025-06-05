import gymnasium as gym

import numpy as np
import os
from stable_baselines3 import PPO

import KiwiGym_createEnv_v4F

import matplotlib.pyplot as plt

# %%
if __name__=="__main__":

    reward_acc=[]

    load_dir = "saved_models/ppo_agent_4F"
    model_name="ppo_agent_4F"
    model=PPO.load(os.path.join(load_dir,model_name),device="cpu") 

    
    for i in range(1):
        env = gym.make('kiwiGym-v4F') 
        obs,_=env.reset()    

        while(True):
            action, _ = model.predict(obs,deterministic=True)  

            print(action)
            obs, reward, terminated, _, _ = env.step(action)

            if(terminated):
                env.render()
                reward_acc.append(reward)
                break
# %%
        mbr=0 #0-2
        species=4 #0-4
        
        if species !=3:
            tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sample']
        else:
            tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sensor']
        
        
        XX=env.unwrapped.kiwiGym.XX['sample'][mbr][species]
        
        tp=env.unwrapped.kiwiGym.DD_historic[mbr]['time_pulse']
        Fp=env.unwrapped.kiwiGym.DD_historic[mbr]['Feed_pulse']
        
        plt.plot(tt,XX,'.')
        plt.show()
        plt.plot(tp,Fp)
