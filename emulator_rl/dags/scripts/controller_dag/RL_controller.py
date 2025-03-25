import json
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
import sys


# function to parse data from ilab database to neural network input
# %%
def create_input_from_db(row_mbrs, db_output, mbr_actions):

    iter = 4

    # create vector with 3 MBR action elements
    actions = np.transpose(np.array([mbr_actions[mbr] for mbr in row_mbrs]))

    e_vector = [16]
    d_vector = actions + np.tile([0, 0, 0], 12 - len(actions))
    y_vector = []

    # add all the measurements to the vector
    # for mbr in row_mbrs:        
    #     for measurement in ["OD600", "Glucose", "Acetate", "DOT", "Fluo_RFP"]:
            # for something in hour (30 DOT measures in one hour)
                # if measurement == "DOT":
                    # get min value in the hour


    # normalize

    # np.tile([20,10,10]+[105]*1+[200e3],(self.kiwiGym.time_final)*self.kiwiGym.number_mbr)

    return vector

# ----------------------------------------------
   
db_output = json.load(open("db_output.json", "r"))
feed_ref = json.load(open("feed_reference.json", "r"))
mbrs_actions = json.load(open("actions.json", "r"))

# %%
mbrs = np.arange(19419, 19443).reshape(8, 3, order='F')

model = PPO.load("model", print_system_info=True, env=None)

action_values = np.arange(-5, 5.5, 0.5) # Is it better to load from env?

new_feed = {}

# iterate over MBRs rows
for row_mbrs in mbrs:

    # preprocess data from JSON to network input
    vector_input = create_input_from_db(row_mbrs, db_output, mbrs_actions)

    # load model and predict actions per row    
    actions = model.predict(vector_input)

    # from predictions calculate feeding and update from reference profile
    for idx, action in enumerate(actions):
        # TODO: sumarlo solo a la hora correspondiente
        new_feed[row_mbrs[idx]] = feed_ref[row_mbrs[idx]] + action_values[action]

        # update mbr actions
        if row_mbrs[idx] in mbrs_actions:
            mbrs_actions[row_mbrs[idx]].append(action)
        else:
            mbrs_actions[row_mbrs[idx]] = [action]

    # TODO: check some min and max volume restrictions

# save updated profile to JSON            
with open('feed.json', "w") as json_file:
    json.dump(new_feed, json_file)   

# save updated profile to JSON            
with open('actions.json', "w") as json_file:
    json.dump(mbrs_actions, json_file)  

