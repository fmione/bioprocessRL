import json
import numpy as np
# from stable_baselines3 import PPO
# import gymnasium as gym
# import KiwiGym_createEnv_v4

# %% 
   
with open('db_output.json') as json_file: 
    db_output = json.load(json_file)

with open('feed_reference.json') as json_file:
    feed_ref = json.load(json_file)

nn_input = open('nn_input.pkl', "r").read()


# %%
mbr_rows = np.arange(19419, 19443).reshape(8, 3, order='F')


# for row in mbr_rows:

# for measurement in ["OD600", "Glucose", "Acetate", "DOT", "Fluo_RFP"]:


# preprocess data from JSON to network input


# load model and predict actions per row


# %%
model_name = "lr_0.0005_ns_110_bs_55_cp_True"
env = gym.make("kiwiGym-v4")
model = PPO.load(f"saved_models/ppo_env4/{model_name}", print_system_info=True, env=env)


# from predictions calculate feeding and update from reference profile


# save updated profile to JSON


            
with open('feed.json', "w") as json_file:
    json.dump(feed_updated, json_file)
         
