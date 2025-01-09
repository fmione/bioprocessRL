# ***************************************************************** #
#   THIS FILE MODELS THE ENVIRONMENT USING THE PROBLEM DEFINITION   #
# ***************************************************************** #

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from bioprocess import Bioprocess, FeedAction


register(
    id="BioprocessEnv-v0",
    entry_point="environment:BioprocessEnv"
)

class BioprocessEnv(gym.Env):
     
     def __init__(self):
        # check spaces.Sequence for a sequence of Discrete actions.
        self.action_space = spaces.Discrete(len(FeedAction))
        self.observation_space = spaces.Box()

        self.reset()        
     
     def _get_obs(self):
         pass
     
     def reset(self):
         pass
     
     def step(self, action):
         pass
     
