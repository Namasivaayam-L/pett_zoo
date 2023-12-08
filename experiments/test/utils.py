import configparser
def load_config(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)

    env = {}
    env["net_file"] = config["Sumo"]["net_file"]
    env["route_file"] = config["Sumo"]["route_file"]
    env["out_csv_name"] = config["Sumo"]["out_csv_name"]
    env["single_agent"] = config["Sumo"].getboolean("single_agent")
    env["use_gui"] = config["Sumo"].getboolean("use_gui")
    env["num_seconds"] = int(config["Sumo"]["num_seconds"])
    env['yellow_time'] = int(config["Sumo"]["yellow_time"])
    env['min_green'] = int(config["Sumo"]["min_green"])
    env['max_green'] = int(config["Sumo"]["max_green"])
    model = {}
    model["policy"] = config["Model"]["policy"]
    model["learning_rate"] = float(config["Model"]["learning_rate"])
    try:
        model["learning_starts"] = int(config["Model"]["learning_starts"])
        model["buffer_size"] = int(config["Model"]["buffer_size"])
        model["train_freq"] = int(config["Model"]["train_freq"])
        model["target_update_interval"] = int(config["Model"]["target_update_interval"])
        model["exploration_fraction"] = float(config["Model"]["exploration_fraction"])
        model["exploration_final_eps"] = float(config["Model"]["exploration_final_eps"])
    except:
        pass
    model["verbose"] = int(config["Model"]["verbose"])

    train = {}
    train["total_timesteps"] = int(config["Train"]["total_timesteps"])
    train["progress_bar"] = config["Train"]["progress_bar"].lower() == "true"
    train["n_eval_episodes"] = int(config["Train"]["n_eval_episodes"])

    return env, model, train


# import configparser
# import sys
# import gymnasium as gym
# from stable_baselines3.dqn.dqn import DQN
# import traci
# from sumo_rl import SumoEnvironment

# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")


# if __name__ == "__main__":
#     # Load parameters from config.ini file
#     config = configparser.ConfigParser()
#     config.read('config.ini')

#     env_name = config['DEFAULT']['env_name']
#     learning_rate = float(config['Model']['learning_rate'])
#     learning_starts = int(config['Model']['learning_starts'])
#     buffer_size = int(config['Model']['buffer_size'])
#     train_freq = int(config['Model']['train_freq'])
#     target_update_interval = int(config['Model']['target_update_interval'])
#     exploration_fraction = float(config['Model']['exploration_fraction'])
#     exploration_final_eps = float(config['Model']['exploration_final_eps'])
#     verbose = int(config['Model']['verbose'])

#     total_timesteps = int(config['TRAINING']['total_timesteps'])
#     progress_bar = config['TRAINING']['progress_bar'].lower() == 'true'
#     n_eval_episodes = int(config['TRAINING']['n_eval_episodes'])

#     # Load SumoEnvironment parameters from config.ini file
#     net_file = config['SUMO_RL']['net_file']
#     route_file = config['SUMO_RL']['route_file']
#     out_csv_name = config['SUMO_RL']['out_csv_name']
#     single_agent = config['SUMO_RL']['single_agent'].lower() == 'true'
#     use_gui = config['SUMO_RL']['use_gui'].lower() == 'true'
#     num_seconds = int(config['SUMO_RL']['num_seconds'])

#     # Define the environment
#     env = SumoEnvironment(
#         net_file=net_file,
#         route_file=route_file,
#         out_csv_name=out_csv_name,
#         single_agent=single_agent,
#         use_gui=use_gui,
#         num_seconds=num_seconds,
#     )

#     # Create the Model model using parameters from config.ini file
#     model = DQN(
#         env=env,
#         policy="MlpPolicy",
#         learning_rate=learning_rate,
#         learning_starts=learning_starts,
#         buffer_size=buffer_size,
#         train_freq=train_freq,
#         target_update_interval=target_update_interval,
#         exploration_fraction=exploration_fraction,
#         exploration_final_eps=exploration_final_eps,
#         verbose=verbose,
#     )

#     # Train the Model model
#     model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

#     # Evaluate the trained model
#     mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
#     print(f"Mean reward over 10 evaluation episodes: {mean_reward}")

#     # Save the trained model
#     model.save('models/Model-big-intx')
