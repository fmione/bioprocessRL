import gymnasium as gym

import numpy as np

import KiwiGym_createEnv_v5C

import matplotlib.pyplot as plt
# %%

if __name__=="__main__":
    env = gym.make('kiwiGym-v5C') 
    obs,_=env.reset()
    # print(obs[0:11])
    

    while(True):
    
        rand_action =0#env.action_space.sample()#
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(rand_action)
            

        if(terminated):
            env.render()
                
            # obs = env.reset()[0]
            break

        
