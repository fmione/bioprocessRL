import gymnasium as gym

import numpy as np

import KiwiGym_createEnv_v3
# %%

if __name__=="__main__":
    env = gym.make('kiwiGym-v3') 
    env.reset()
    # Design variables
    number_mbr=env.unwrapped.kiwiGym.number_mbr
    numb_iter=env.unwrapped.kiwiGym.time_final
    number_samples=4+25
    number_action=1
    
    II_obs={}    
    I_obs={}
    # I_actions=np.zeros(numb_iter*number_mbr*(number_action))
    I_rewards={}
    
    Iter=0
    Iter_step=0
    index_samples=np.arange(number_mbr*(number_samples))
    index_actions=np.arange(number_mbr*(number_action))
    while(True):

        rand_action =[i for i in env.action_space.sample()]#env.action_space.sample().tolist()#[10,10,10]#env.unwrapped.action_values[env.action_space.sample()]
        obs, reward, terminated, _, _ = env.step(rand_action)
        # print(reward,rand_action,obs[0:15])
        
        I_obs[Iter_step]=obs
        # I_actions[index_actions+len(index_actions)*(Iter_step-1)]=np.array(rand_action)
        I_rewards[Iter_step]=reward

        Iter_step=Iter_step+1
        if(terminated):
            # break
            env.render()
            
            obs = env.reset()[0]
            II_obs[Iter]={'obs':I_obs,'reward':I_rewards}
            
            I_obs={}
            # I_actions=np.zeros(numb_iter*number_mbr*(number_action))
            I_rewards={}
            Iter_step=0
            Iter=Iter+1
            
            