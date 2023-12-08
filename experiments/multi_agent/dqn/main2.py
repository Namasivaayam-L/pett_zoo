import os,sys,configparser,time,shutil,pandas as pd
import dqn,memory

from sumo_rl import parallel_env
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

time = time.strftime('%H:%M:%S_%d-%m-%Y')

ini_file = "experiments/multi_agent/dqn/config2.ini"
config = configparser.ConfigParser()
config.read(ini_file)

output_path = config["Sumo"]['out_dir']
model_path = output_path+'models/'
log_path = output_path+'logs/'
fine_tune_model_path = None
if config["Model"].getboolean('fine_tune'):
    fine_tune_model_path = config["Model"]['fine_tune_model_path']

batch_size, gamma, learning_rate  = config["Model"].getint("batch_size"),config["Model"].getfloat("gamma"),config["Model"].getfloat("learning_rate")
num_episodes,buffer_size = config["Model"].getint('num_episodes'),config["Memory"].getint("buffer_size")
num_layers, width  = config["Model"].getint("num_layers"), config["Model"].getint("width")
epsilon, min_epsilon, decay = config["Model"].getfloat("epsilon"),config["Model"].getfloat("min_epsilon"),config["Model"].getfloat("decay")
os.makedirs(output_path,exist_ok=True)
os.makedirs(model_path,exist_ok=True)
shutil.copyfile(ini_file,output_path+'config.ini')

env = parallel_env(
    net_file=config["Sumo"]['net_file'],
    route_file=config["Sumo"]['route_file'],
    out_csv_name=output_path,
    use_gui=config["Sumo"].getboolean('use_gui'),
    num_seconds=int(config["Sumo"]['num_seconds']),
    yellow_time=int(config["Sumo"]['yellow_time']),
    min_green=int(config["Sumo"]['min_green']),
    max_green=int(config["Sumo"]['max_green']),
    reward_fn=config["Sumo"]['reward_fn']
)

step,metrics,info = 0,[],{"step":0}
agents = {ts: dqn.DQN(ts, env.action_spaces[ts].n, env.observation_spaces[ts].shape[0], width, num_layers, batch_size, gamma, learning_rate, model_path, fine_tune_model_path) for ts in env.possible_agents}
experience_replay = memory.Memory(buffer_size)
for ep in range(num_episodes):
    info['ep'] = ep
    state,infos = env.reset()
    terminations = {a: False for a in agents}
    while not all(terminations.values()):
        actions = { ts: agents[ts].act(state[ts], epsilon) for ts in env.possible_agents} 
        next_state, rewards,  terminations, truncations, infos= env.step(actions)
        info.update(rewards)
        metrics.append(info.copy())
        if all(terminations.values()):
            for key in agents.keys():
                agents[key].learn(experience_replay.sample_batch(batch_size))
            break
        for ts in env.possible_agents:
            experience_replay.store(state[ts], actions[ts], rewards[ts], next_state[ts])
        state = next_state
        info['step']+=1
    epsilon = max(epsilon * decay, min_epsilon)
    env.save_csv(output_path, ep)
    env.close()
df = pd.DataFrame(metrics)
df.to_csv(output_path+f'rewards.csv',index=False)