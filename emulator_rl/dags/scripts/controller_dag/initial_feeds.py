import json
import numpy as np

config = json.load(open("config.json", "r"))

# convert to cummulative feed and assing feed profile to each MBR group
feed = {exp_id: {} for exp_id in config["exp_ids"]}
for mbr_row in config["mbr_groups"]:
    for i, mbr in enumerate(mbr_row):
        feed[mbr]['measurement_time'] = (np.array(config["feed_pulses_reference"][i]["time_pulse"]) * 3600).astype(int).tolist()
        feed[mbr]['setpoint_value'] = np.cumsum(config["feed_pulses_reference"][i]["Feed_pulse"]).tolist()

with open('feed.json', "w") as outfile:
    json.dump(feed, outfile) 