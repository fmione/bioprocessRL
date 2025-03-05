import gymnasium as gym

import numpy as np

import KiwiGym_createEnv_v5C

import matplotlib.pyplot as plt
# %%

if __name__=="__main__":
    env = gym.make('kiwiGym-v5C') 
    obs,_=env.reset()
    # print(obs[0:11])
  
    II_obs={}    
    I_obs={}
    I_rewards={}
    
    Iter=0
    Iter_step=0
    reward_collected=[]
    
    for i in np.linspace(0.5,1,41):
        while(True):
    
            rand_action =i#.6#env.action_space.sample()#
            obs, reward, terminated, _, _ = env.step(rand_action)
            print(i)
            # print(rand_action)
            
            I_obs[Iter_step]=obs
            I_rewards[Iter_step]=reward
    
            Iter_step=Iter_step+1
            if(terminated):
                reward_collected.append(reward)
                # env.render()
                
                obs = env.reset()[0]
                break
                II_obs[Iter]={'obs':I_obs,'reward':I_rewards}
                
                I_obs={}
                I_rewards={}
                Iter_step=0
                Iter=Iter+1
        
    plt.plot(reward_collected)