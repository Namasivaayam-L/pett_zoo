import os
import sys
import configparser
import dqn
import memory
import multi_dqn
from sumo_rl import SumoEnvironment
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

ini_file = "experiments/multi_agent/dqn2/config.ini" 
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

batch_size, gamma, learning_rate  = config["Model"].getint("batch_size"),config["Model"].getfloat("gamma"),config["Model"].getfloat("learning_rate")
num_episodes,buffer_size, eps = config["Model"].getint('num_episodes'),config["Memory"].getint("buffer_size"),config["Model"].getfloat("epsilon")
num_layers, width = config["Model"].getint("num_layers"), config["Model"].getint("width")
agents =[ dqn.DQN(env.action_space.n, env.observation_space.shape[0], width, num_layers, batch_size, gamma, learning_rate)]
experience_replay = memory.Memory(buffer_size)
ts = 0 
for ep in range(num_episodes):
    state = env.reset()[0]
    done = {"__all__": False}
    while not done["__all__"]:
        # print('state',state)
        actions = agents[ts].act(state, experience_replay, ts, eps)
        next_state, rewards, done, _, _ = env.step(action=actions)
        experience_replay.store(state, actions, rewards, next_state)
        state = next_state
        print(done)
        # if done:
        agents[ts].learn(experience_replay.sample_batch(batch_size,ts))
    eps -= 0.09
    env.save_csv('outputs/multi_agent/dqn/dqn', ep)
    env.close()