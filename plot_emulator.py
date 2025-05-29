import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_emulator_training():
    sns.set_theme(style="darkgrid")
    
    with open("db_output.json", "r") as f:
        data = json.load(f)

    for species, sp_name in enumerate(["OD600", "Glucose", "Acetate", "DOT", "Fluo_RFP"]):
        
        for mbr in data:
            df = pd.DataFrame({"time": data[mbr]["measurements_aggregated"][sp_name]["measurement_time"], "value": data[mbr]["measurements_aggregated"][sp_name][sp_name]})

            # Convert OD600 to biomass
            if sp_name == "OD600":                
                df["value"] = df["value"] / 2.7027

            # Convert time to hours
            df["time"] = df["time"] / 3600

            # lines
            plt.plot(df["time"], df["value"])
            # points
            # plt.plot(df["time"], df["value"], ".")

        if sp_name == "DOT":
            plt.ylim(0, 105)
            plt.axhline(y=20, color="#A8A5A5", linestyle='--')
            plt.text(x=0.2, y=20 + 1.5, s="DOT constraint",   color='#A8A5A5', fontsize=9)

        plt.xlabel(f"Time $[h]$", fontweight='bold')
        plt.ylabel(f"{sp_name}", fontweight='bold')
        plt.tight_layout()
        # plt.show()

        os.makedirs(os.path.dirname("plots/plots_emulator/"), exist_ok=True)
        plt.savefig(f"plots/plots_emulator/{sp_name}.png", dpi=600)
        plt.clf()

    


plot_emulator_training()