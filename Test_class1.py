import numpy as np
import gymnasium as gym
import KiwiGym_createEnv_v1
# %%

if __name__=="__main__":
    env = gym.make('kiwiGym-v1') 
    env.reset()
    # Design variables
    number_mbr=env.unwrapped.kiwiGym.number_mbr
    numb_iter=env.unwrapped.kiwiGym.time_final
    number_samples=4+25
    number_action=1
    
    II_obs={}    
    I_obs=np.zeros(numb_iter*number_mbr*(number_samples+number_action))
    I_actions=np.zeros(numb_iter*number_mbr*(number_action))
    I_rewards=np.zeros(numb_iter)
    
    Iter=0
    Iter_step=0
    index_samples=np.arange(number_mbr*(number_samples))
    index_actions=np.arange(number_mbr*(number_action))
    while(True):

        rand_action =[10,10,10]#env.action_space.sample().tolist()#[10,10,10]#env.unwrapped.action_values[env.action_space.sample()]
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward,rand_action)
        
        I_obs[index_samples+len(index_samples)*(Iter_step-1)]=obs
        I_actions[index_actions+len(index_actions)*(Iter_step-1)]=np.array(rand_action)
        I_rewards[Iter_step]=reward

        Iter_step=Iter_step+1
        if(terminated):
            env.render()
            
            TH_param_mean=np.array([1.2578,0.43041, 0.6439,  2.2048,  0.4063,  0.1143,  0.1848,    287.74,    1.586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[850]*3+[90]*3)
            TH_param=TH_param_mean*(1+0.05*(np.random.random(len(TH_param_mean))-.5)*2)
            
            obs = env.reset(options=TH_param)[0]
            II_obs[Iter]={'obs':I_obs.copy(),'action':I_actions.copy(),'reward':I_rewards.copy()}
            
            I_obs=np.zeros(numb_iter*number_mbr*(number_samples+number_action))
            I_actions=np.zeros(numb_iter*number_mbr*(number_action))
            I_rewards=np.zeros(numb_iter)
            Iter_step=0
            Iter=Iter+1