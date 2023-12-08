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
from stable_baselines3 import Rainbow
from stable_baselines3.common.evaluation import evaluate_policy
from sumo_rl import SumoEnvironment
from utils import load_config

ini_file = "config.ini"  # Replace with the actual path to your INI file

# Load parameters from config.ini file
env_params, model_params, training_params = load_config(ini_file)

env = SumoEnvironment(**env_params)
# Rainbow model
model = Rainbow(
    env=env,
    **model_params,
)

model.learn(total_timesteps=training_params["total_timesteps"], progress_bar=training_params["progress_bar"])
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=training_params["n_eval_episodes"])
print(f"Mean reward over 10 evaluation episodes: {mean_reward}")
model_save_path = "models/" + env_params["out_csv_name"][8:]
model.save(model_save_path)