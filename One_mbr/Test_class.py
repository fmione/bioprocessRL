import gymnasium as gym

import numpy as np

import KiwiGym_createEnv_v5
# %%

if __name__=="__main__":
    env = gym.make('kiwiGym-v5') 
    obs,_=env.reset()
    # print(obs[0:11])
  
    II_obs={}    
    I_obs={}
    I_rewards={}
    
    Iter=0
    Iter_step=0

    while(True):

        rand_action =env.action_space.sample()#env.action_space.sample().tolist()#[10,10,10]#env.unwrapped.action_values[env.action_space.sample()]
        obs, reward, terminated, _, _ = env.step(rand_action)
        # print(obs[0:11])
        print(rand_action)
        
        I_obs[Iter_step]=obs
        I_rewards[Iter_step]=reward

        Iter_step=Iter_step+1
        if(terminated):
            
            env.render()
            break
            obs = env.reset()[0]
            II_obs[Iter]={'obs':I_obs,'reward':I_rewards}
            
            I_obs={}
            I_rewards={}
            Iter_step=0
            Iter=Iter+1
            
            