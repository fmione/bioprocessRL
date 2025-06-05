import gymnasium as gym

import numpy as np
import os
from stable_baselines3 import PPO

import KiwiGym_createEnv_v4F


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# %%
if __name__=="__main__":


    reward_opt=[]
    reward_base=[]
    reward_zero=[]
    
    load_dir = "saved_models/ppo_agent_4F"
    model_name="ppo_agent_4F"
    model=PPO.load(os.path.join(load_dir,model_name),device="cpu")
    
    load_dir_base = "saved_models/ppo_agent_4F"
    model_name_base="ppo_agent_4F_0"
    model_base=PPO.load(os.path.join(load_dir_base,model_name_base),device="cpu")
    #######
    for i in range(10):
        print("iter: ", i)
        env = gym.make('kiwiGym-v4F') 
        obs,_=env.reset()    
        TH_env=env.unwrapped.kiwiGym.TH_param

        while(True):
            action, _ = model.predict(obs,deterministic=True)  
            print(action)
            obs, reward, terminated, _, _ = env.step(action)
            
            if(terminated):
                reward_opt.append(reward)
                break
        #######
        obs,_=env.reset() 
        env.unwrapped.kiwiGym.TH_param=TH_env
        while(True):
            action_base, _ = model_base.predict(obs,deterministic=True)  
            model_base
            obs, reward, terminated, _, _ = env.step(action_base)
            print(action)
            if(terminated):
                print("########")
                reward_base.append(reward)
                break      
        #######
        obs,_=env.reset() 
        env.unwrapped.kiwiGym.TH_param=TH_env
        while(True):
            obs, reward, terminated, _, _ = env.step([10,10,10])
            if(terminated):
                print("########")
                reward_zero.append(reward)
                break  
    print("reward_opt: ",reward_opt,"reward_base: ",reward_base,"reward_zero: ", reward_zero)
    print(" ")
    print("reward_opt_mean: ",np.mean(reward_opt),"reward_base_mean: ",np.mean(reward_base),"reward_zero_mean: " ,np.mean(reward_zero))