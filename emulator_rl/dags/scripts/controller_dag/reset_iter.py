import json

config = json.load(open("config.json", "r"))
config["iter"] = 0
config["mbrs_actions"] = {exp_id: [] for exp_id in config["exp_ids"]}
json.dump(config, open("config.json", "w"))