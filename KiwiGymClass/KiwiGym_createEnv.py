'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np

from kiwiGym import kiwiGym
# %%
# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='kiwiGym-v1',                                # call it whatever you want
    entry_point='KiwiGym_createEnv:kiwiGymEnv', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class kiwiGymEnv(gym.Env):
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
        self.action_space = spaces.MultiDiscrete([len(self.action_values)] * self.kiwiGym.number_mbr)  # Actions are indices
        
        # Gym requires defining the observation space. The observation space consists of the robot's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().

        self.observation_space = spaces.Box(
            low=np.zeros(29*self.kiwiGym.number_mbr*round(self.kiwiGym.time_final)),
            high=np.tile([30,30,30]+[105]*25+[200e3],self.kiwiGym.number_mbr*round(self.kiwiGym.time_final)),
            dtype=np.float64
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        if options is None:
            TH_reset=[]
        else:
            TH_reset=options

        # Reset the env. 
        self.kiwiGym.reset(seed=seed,TH_param=TH_reset)

        # Construct the observation state:
        obs = self.kiwiGym.obs
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        action_val = [self.action_values[i] for i in action]
        obs,reward,terminated = self.kiwiGym.perform_action(action_val)

        # Additional info to return. 
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            print('Action: ',action_val)
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info

    # Gym required function to render environment
    def render(self):
        self.kiwiGym.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('kiwiGym-v1')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):

        rand_action =[8,10,12]#env.action_space.sample()#env.unwrapped.action_values[env.action_space.sample()]
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward,obs[-29],rand_action)

        if(terminated):
            TH_param=np.array([1.2578*(np.random.random(1)[0]-.5)/2, 0.43041, 0.6439,  2.2048,  0.5063,  0.1143,  0.1848,    287.74,    1.2586, 1.5874,  0.3322,  0.0371,  0.0818,  7.0767,  0.4242, .1057]+[750]*3+[90]*3)

            obs = env.reset(options=TH_param)[0]