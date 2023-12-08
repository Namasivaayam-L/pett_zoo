import os
import sys
import gym
import configparser
import shutil

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sumo_rl import SumoEnvironment

ini_file = "experiments/single_agent/dqn/config.ini" 
config = configparser.ConfigParser()
# Load parameters from config.ini file
config.read(ini_file)

env = SumoEnvironment(
    net_file=config["Sumo"]['net_file'],
    single_agent=config["Sumo"].getboolean('single_agent'),
    route_file=config["Sumo"]['route_file'],
    out_csv_name=config["Sumo"]['out_csv_name'],
    use_gui=config["Sumo"].getboolean('use_gui'),
    num_seconds=int(config["Sumo"]['num_seconds']),
    yellow_time=int(config["Sumo"]['yellow_time']),
    min_green=int(config["Sumo"]['min_green']),
    max_green=int(config["Sumo"]['max_green']),
)

# DQN model
model = DQN(
    env=env,
    batch_size=int(config['Model']['batch_size']),
    policy = config['Model']['policy'],
    learning_rate=float(config["Model"]["learning_rate"]),
    learning_starts=float(config["Model"]["learning_starts"]),
    buffer_size=int(config["Model"]["buffer_size"]),
    train_freq=int(config["Model"]["train_freq"]),
    target_update_interval=int(config["Model"]["target_update_interval"]),
    exploration_fraction=float(config["Model"]["exploration_fraction"]),
    exploration_final_eps=float(config["Model"]["exploration_final_eps"]),
    gamma=float(config['Model']['gamma']),
    device=config['Model']['device'],
    tensorboard_log=config['Model']['tensorboard_log'],
    verbose=int(config["Model"]["verbose"]),
)
model.learn(total_timesteps=int(config["Train"]["total_timesteps"]), progress_bar=config["Train"].getboolean('progress_bar'))
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=int(config["Train"]["n_eval_episodes"]))
print(f"Mean reward over 10 evaluation episodes: {mean_reward}")
model_save_path = "models/" + config["Sumo"]["out_csv_name"][8:]
os.makedirs(model_save_path,exist_ok=True)
model.save(model_save_path)

shutil.copyfile('experiments/'+config["Model"]["name"]+'/config.ini', config["Sumo"]['out_csv_dir'])