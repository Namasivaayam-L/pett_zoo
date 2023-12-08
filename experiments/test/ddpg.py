import os
import sys
import gym
import configparser

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import traci
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from sumo_rl import SumoEnvironment
from utils import load_config

ini_file = "experiments/config.ini"  # Replace with the actual path to your INI file

# Load parameters from config.ini file
env, model, training = load_config(ini_file)

env = SumoEnvironment(
    net_file=env['net_file'],
    single_agent=env['single_agent'],
    route_file=env['route_file'],
    out_csv_name=env['out_csv_name'],
    use_gui=env['use_gui'],
    num_seconds=env['num_seconds'],
    yellow_time=env['yellow_time'],
    min_green=env['min_green'],
    max_green=env['max_green'],
)
#DDPG model
model = DDPG(
    env=env,
    policy="MlpPolicy",
    # learning_rate=model["learning_rate"],
    # learning_starts=model["learning_starts"],
    # buffer_size=model["buffer_size"],
    # train_freq=model["train_freq"],
    verbose=model["verbose"],
)

model.learn(total_timesteps=training["total_timesteps"], progress_bar=training["progress_bar"])
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=training["n_eval_episodes"])
print(f"Mean reward over 10 evaluation episodes: {mean_reward}")
model_save_path = "models/+" + "-" + env["out_csv_name"].split("/")[1]
model.save(model_save_path)
