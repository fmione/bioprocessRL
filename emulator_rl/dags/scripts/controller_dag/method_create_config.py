import numpy as np
import json

config = dict()

# MBRs definition
config["runID"] = 623
config["exp_ids"] = np.arange(19419, 19443).tolist()
config["number_mbr"] = 3
config["mbr_groups"] = np.array(config["exp_ids"]).reshape(8, 3, order='F').tolist()

# DAG structure in minutes
config["time_start_checking_db"] = 5
config["time_bw_check_db"] = 1

# other variables
config["iter"] = 0
config["time_batch"] = 5
config["time_final"] = 16

config["action_values"] = np.arange(-5, 5.5, 0.5).tolist()
config["mu_reference"] = [0.15253191, 0.12974431, 0.1024308]

# init mbrs actions
config["mbrs_actions"] = {exp_id: [] for exp_id in config["exp_ids"]}

# create feeding pulses
time_pulses = np.arange(config["time_batch"] + 5/60, config["time_final"], 10/60)
config["feed_pulses_reference"] = []
for i in range(config["number_mbr"]):
    feed_profile_i = (36.33) * config["mu_reference"][i] * np.exp(config["mu_reference"][i] * (time_pulses - time_pulses[0]))
    feed_profile_i[time_pulses >= 10] = (36.33) * config["mu_reference"][i] * np.exp(config["mu_reference"][i] * (10 - time_pulses[0]))
    feed_profile_i = np.round(feed_profile_i * 2) / 2
    feed_profile_i[feed_profile_i < 5] = 5

    config["feed_pulses_reference"].append({'time_pulse': time_pulses.tolist(), 'Feed_pulse': feed_profile_i.tolist()})

# save config file
with open('config.json', "w") as outfile:
    json.dump(config, outfile) 