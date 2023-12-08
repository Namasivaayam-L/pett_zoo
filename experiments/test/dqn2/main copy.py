import os,sys,configparser,time,shutil
import dqn,memory

from sumo_rl import parallel_env
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

time = time.strftime('%H:%M:%S_%d-%m-%Y')

ini_file = "experiments/multi_agent/dqn/config.ini"
config = configparser.ConfigParser()
config.read(ini_file)

output_path = config["Sumo"]['out_dir']
model_path = output_path+'model/model.keras'
csv_path = output_path+config["Model"]['Name']
log_path = output_path+'logs/'
fine_tune_model_path = None
if config["Model"].getboolean('fine_tune'):
    fine_tune_model_path = config["Model"]['fine_tune_model_path']

batch_size, gamma, learning_rate  = config["Model"].getint("batch_size"),config["Model"].getfloat("gamma"),config["Model"].getfloat("learning_rate")
num_episodes,buffer_size = config["Model"].getint('num_episodes'),config["Memory"].getint("buffer_size")
num_layers, width  = config["Model"].getint("num_layers"), config["Model"].getint("width")
epsilon, min_epsilon, decay = config["Model"].getfloat("epsilon"),config["Model"].getfloat("min_epsilon"),config["Model"].getfloat("decay")
os.makedirs(output_path,exist_ok=True)
os.makedirs(log_path,exist_ok=True)
os.makedirs(output_path+'model/',exist_ok=True)
shutil.copyfile(ini_file,output_path+'config.ini')

env = parallel_env(
    net_file=config["Sumo"]['net_file'],
    route_file=config["Sumo"]['route_file'],
    out_csv_name=csv_path,
    use_gui=config["Sumo"].getboolean('use_gui'),
    num_seconds=int(config["Sumo"]['num_seconds']),
    yellow_time=int(config["Sumo"]['yellow_time']),
    min_green=int(config["Sumo"]['min_green']),
    max_green=int(config["Sumo"]['max_green']),
    reward_fn=config["Sumo"]['reward_fn']
)


with open(log_path+time+'.txt','w') as log_file:
    log_file.write('step,ep,r1,r2,r3,r4,l')
    agents = {ts: dqn.DQN(env.action_spaces[ts].n, env.observation_spaces[ts].shape[0], width, num_layers, batch_size, gamma, learning_rate, model_path, log_file, fine_tune_model_path) for ts in env.possible_agents}
    experience_replay = memory.Memory(buffer_size)
    for ep in range(num_episodes):
        print(f'==== EPISODE : {ep} ==== EPSILON : {epsilon} ====',file=log_file)
        state,infos = env.reset()
        terminations = {a: False for a in agents}
        while not all(terminations.values()):
            print(f'========================================================================================================',file=log_file)
            actions = { ts: agents[ts].act(state[ts], epsilon) for ts in env.possible_agents} 
            next_state, rewards,  terminations, truncations, infos= env.step(actions)
            # for ts,action in actions.items():
            #     print(f'Ts:{ts}, action:{action}',file=log_file)
            print(f'rewards:{rewards}',file=log_file)
            for ts in env.possible_agents:
                try: experience_replay.store(state[ts], actions[ts], rewards[ts], next_state[ts])
                except: pass
            state = next_state
            if all(terminations.values()):
                for key in agents.keys():
                    print(f'========================================================================================================',file=log_file) 
                    print(f'ts-{key}: learning....',file=log_file)
                    # print(f'========================================================================================================',file=log_file) 
                    agents[key].learn(experience_replay.sample_batch(batch_size))
                # print(f'========================================================================================================',file=log_file) 
        epsilon = max(epsilon * decay, min_epsilon)
        env.save_csv(csv_path, ep)
        env.close()