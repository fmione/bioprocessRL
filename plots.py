from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import KiwiGym_createEnv_v4F


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Plot training data (two steps) and the fitted curve
def plot_model_training():

    sns.set_theme(style="darkgrid")

    # get event data from TensorBoard
    log_dir_step1 = "logs/ppo_env4/lr_0.0005_ns_110_bs_55_cp_False_1"
    event_step1= EventAccumulator(log_dir_step1)
    event_step1.Reload()

    log_dir_step2 = "logs/ppo_env4/2nd_lr_0.0001_lr_0.0005_ns_110_bs_55_cp_False_1"
    event_step2= EventAccumulator(log_dir_step2)
    event_step2.Reload()

    # get episode mean rewards
    reward1 = event_step1.Scalars("rollout/ep_rew_mean")
    reward_df1 = pd.DataFrame([(e.step, e.value) for e in reward1], columns=["step", "reward"])
    reward2 = event_step2.Scalars("rollout/ep_rew_mean")
    reward_df2 = pd.DataFrame([(e.step, e.value) for e in reward2], columns=["step", "reward"])

    # Plot training data
    _, ax = plt.subplots()
    ax.plot(reward_df1["step"], reward_df1["reward"], label="Train Stage 1")
    ax.plot(reward_df2["step"] + reward_df1["step"].iloc[-1], reward_df2["reward"], label="Train Stage 2")

    # Plot Fit curve
    reward_df2_ext = reward_df2.copy()
    reward_df2_ext["step"] = reward_df2_ext["step"] + reward_df1["step"].iloc[-1]
    df_concat = pd.concat([reward_df1, reward_df2_ext], ignore_index=True)

    coef = np.polyfit(df_concat["step"], df_concat["reward"], deg=20)
    poly = np.poly1d(coef)

    x_fit = np.linspace(df_concat["step"].min(), df_concat["step"].max(), 100)
    y_fit = poly(x_fit)

    ax.plot(x_fit, y_fit, label="Fit", linestyle='--', color='#333333')

    # Set plot labels and legend
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylim(0, 3.5)
    plt.xlabel(f"Steps", fontweight='bold')
    plt.ylabel(f"Mean Reward", fontweight='bold')
    plt.legend(fontsize=9, title="References", title_fontsize=10)
    plt.tight_layout()
    plt.show()
    

# Auxiliar function to get the species from the environment
def aux_get_species_from_env(env):
    result = {}
    for mbr in range(3):
        result[mbr] = {}
        for species in range(5):
            if species !=3:
                tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sample']
            else:
                tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sensor']
            
            result[mbr][species] = {"tt": tt, "X": env.unwrapped.kiwiGym.XX['sample'][mbr][species]}
        
    return result

# Plot the results of the different models for one set of parameters (4F, 4F_0, 4F_no_actions)
def plot_model_comparative():  
    sns.set_theme(style="darkgrid")

    env = gym.make('kiwiGym-v4F')    
    load_dir = "saved_models/ppo_agent_4F"  

    experiments = 100
    models = ["ppo_agent_4F_0", "ppo_agent_4F", "no_agent"]
    results = {model_name: [] for model_name in models}

    for a in range(experiments):
        obs,_ = env.reset() 
        TH_env=env.unwrapped.kiwiGym.TH_param

        for model_name in models:

            if model_name != "no_agent":
                model=PPO.load(os.path.join(load_dir,model_name),device="cpu")

            obs,_ = env.reset()         
            env.unwrapped.kiwiGym.TH_param=TH_env

            while(True):
                if model_name == "no_agent":
                    action = [10, 10, 10]
                else:
                    action, _ = model.predict(obs,deterministic=True)  

                # print(action)
                obs, reward, terminated, _, _ = env.step(action)

                if(terminated):
                    break

            # get results
            results[model_name].append(aux_get_species_from_env(env))
    
    # Plot results
    default_colors = sns.color_palette()
    colors = default_colors[:3]

    for model_name in models:
        for species, sp_name in enumerate(["Biomass", "Glucose", "Acetate", "DOT", "Fluo_RFP"]):
            for it in range(experiments):
                for mbr in range(3):
                    plt.plot(results[model_name][it][mbr][species]["tt"], results[model_name][it][mbr][species]["X"], '.', color=colors[mbr])

            if species ==3:
                plt.ylim(0, 105)
                plt.axhline(y=20, color="#A8A5A5", linestyle='--')
                plt.text(x=0.2, y=20 + 1.5, s="DOT constraint",   color='#A8A5A5', fontsize=9)

            plt.xlabel(f"Time $[h]$", fontweight='bold')
            plt.ylabel(f"{sp_name}", fontweight='bold')
            legend_elements = [Line2D([0], [0], marker='.', color='none', label=f"MBR {i+1}",
                    markerfacecolor=colors[i], markersize=10, markeredgecolor="none") for i in range(3)]
            plt.legend(handles=legend_elements, fontsize=9, title="References", title_fontsize=10)
            plt.tight_layout()
            # plt.show()

            if not os.path.isdir(os.path.dirname("plots_3mbr/")):
                os.makedirs(os.path.dirname("plots_3mbr/"))
            plt.savefig(f"plots_3mbr/{model_name}_{sp_name}.png", dpi=600)
            plt.clf()

# Plot results of 4F model (DOT and pulses)
def plot_model4f_results():
    sns.set_theme(style="darkgrid")

    reward_acc=[]

    load_dir = "saved_models/ppo_agent_4F"
    model_name="ppo_agent_4F"
    model=PPO.load(os.path.join(load_dir,model_name),device="cpu") 

    
    for i in range(1):
        env = gym.make('kiwiGym-v4F') 
        obs,_=env.reset()    

        while(True):
            action, _ = model.predict(obs,deterministic=True)  

            print(action)
            obs, reward, terminated, _, _ = env.step(action)

            if(terminated):
                # env.render()
                reward_acc.append(reward)
                break

    # mbr=0 #0-2
    species=3 #0-4     

    for mbr in range(3):
        if species !=3:
            tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sample']
        else:
            tt=env.unwrapped.kiwiGym.DD_historic[mbr]['time_sensor']
            plt.ylim(0, 105)
            plt.axhline(y=20, color="#A8A5A5", linestyle='--')
            plt.text(x=0.2, y=20 + 1.5, s="DOT constraint",   color='#A8A5A5', fontsize=9)
            plt.ylabel(f"DOT $[\%]$", fontweight='bold')

        plt.plot(tt, env.unwrapped.kiwiGym.XX['sample'][mbr][species], '.', label=f"MBR {mbr + 1}")

    plt.xlabel(f"Time $[h]$", fontweight='bold')
    plt.legend(fontsize=9, title="References", title_fontsize=10)
    plt.tight_layout()
    plt.show()
    

    for mbr in range(3):
        tp=env.unwrapped.kiwiGym.DD_historic[mbr]['time_pulse']
        Fp=env.unwrapped.kiwiGym.DD_historic[mbr]['Feed_pulse']
    
        plt.plot(tp, Fp, label=f"MBR {mbr + 1}")

    plt.ylabel(f"Pulse $[\mu L]$ ", fontweight='bold')
    plt.xlabel(f"Time $[h]$", fontweight='bold')
    plt.xlim(0, 14)
    plt.legend(fontsize=9, title="References", title_fontsize=10)
    plt.tight_layout()
    plt.show()


plot_model_training()
# plot_model_comparative()
# plot_model4f_results()
