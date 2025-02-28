
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np

from kiwiGym5 import kiwiGym
# %%
# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='kiwiGym-v5C',                                # call it whatever you want
    entry_point='KiwiGym_createEnv_v5C:kiwiGymEnv5C', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class kiwiGymEnv5C(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 4}


    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # Initialize the WarehouseRobot problem
        self.kiwiGym=kiwiGym()
        # Gym requires defining the action space. 
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_values = np.arange(-5, 5.5, 0.5)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float64)  # Actions are continuous an scaled, later multiplied by conversion factor

        # Gym requires defining the observation space. The observation space consists of the robot's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        
        e_vector=np.array([16])# no encoding
        
        # TH_ref=[1.2578, 0.43041, 0.6439,  2.2048,  0.4063,  0.1143,  0.1848,    287.74,    1.586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[850]*self.kiwiGym.number_mbr+[90]*self.kiwiGym.number_mbr
        y_vector=np.concatenate((np.tile([20,10,10]+[105]*1+[200e3],self.kiwiGym.number_mbr),np.array([])))
        self.observation_upper_bound=np.concatenate([e_vector,y_vector])
        
        self.observation_space = spaces.Box(
            low=(-1)*np.ones(len(self.observation_upper_bound)),
            high=(1)*np.ones(len(self.observation_upper_bound)),
            dtype=np.float64
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.
        np.random.seed(seed)
        TH_param_mean=np.array([1.2578,0.43041, 0.6439,  2.2048,  0.4063,  0.1143,  0.1848,    287.74,    1.586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[850]*self.kiwiGym.number_mbr+[90]*self.kiwiGym.number_mbr)
        TH_reset=TH_param_mean*(1+0.00*(np.random.random(len(TH_param_mean))-.5)/2)

        # Reset the env. 
        self.kiwiGym.reset(seed=seed,TH_param=TH_reset)

        # Construct the observation state:
        obs = np.ones(len(self.observation_upper_bound))*(-1)

        
        obs[0]=int(self.kiwiGym.time_pulses[0])# No encoding
        
        # Integrate up to time batch
        while self.kiwiGym.time_current<round(self.kiwiGym.time_pulses[0]):

            action_val = 0
            obs_raw,_,_ = self.kiwiGym.perform_action(action_val)
            obs[1:]=obs_raw

        self.obs=obs
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Return observation and info
        obs_corrected=obs/self.observation_upper_bound
        return obs_corrected, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        action_val=action*(self.action_values[-1]-self.action_values[0])/2 #Scale the action to [-5,5]
        
        # action_val=round(2*float(action_val))/2 #rounding
        # print(action_val)
        obs_raw,reward,terminated = self.kiwiGym.perform_action(action_val)
        
        obs=self.obs
        #Take the position of obs_norm and allocate in self.obs
        obs[0]=self.kiwiGym.time_current 

        obs[1:]=obs_raw

        self.obs=obs
        
        # Additional info to return. 
        info = {}

        # Render environment
        if (self.render_mode=='human') & (terminated==True):
            print('Action: ',action_val)
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        obs_corrected=obs/self.observation_upper_bound
        return obs_corrected, reward, terminated, False, info

    # Gym required function to render environment
    def render(self):
        self.kiwiGym.render()

# %% For unit testing
if __name__=="__main__":
    env = gym.make('kiwiGym-v5C')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions

    cnt=0
    while(cnt<3):
        rand_action =env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward,rand_action)

        if(terminated):
            env.render()
            
            obs = env.reset()[0]
        cnt+=1
    

            
