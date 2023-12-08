import os, sys, configparser, time, shutil, pandas as pd
import ppo, replay
from tqdm import tqdm
from sumo_rl import parallel_env

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

time = time.strftime("%H:%M:%S_%d-%m-%Y")

ini_file = "experiments/multi_agent/ppo/config.ini"
config = configparser.ConfigParser()
config.read(ini_file)

output_path = config["Sumo"]["out_dir"]
model_path = output_path + "models/"
log_path = output_path + "logs/"
fine_tune_model_path = None
if config["Model"].getboolean("fine_tune"):
    fine_tune_model_path = config["Model"]["fine_tune_model_path"]

batch_size, gamma, learning_rate = (
    config["Model"].getint("batch_size"),
    config["Model"].getfloat("gamma"),
    config["Model"].getfloat("learning_rate"),
)
num_episodes, buffer_size = config["Model"].getint("num_episodes"), config["Memory"].getint("buffer_size")
num_layers, width = config["Model"].getint("num_layers"), config["Model"].getint("width")
gamma, policy_clip, gae_lambda = (
    config["Model"].getfloat("gamma"),
    config["Model"].getfloat("policy_clip"),
    config["Model"].getfloat("gae_lambda"),
)

os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
shutil.copyfile(ini_file, output_path + "config.ini")

env = parallel_env(
    net_file=config["Sumo"]["net_file"],
    route_file=config["Sumo"]["route_file"],
    out_csv_name=output_path,
    use_gui=config["Sumo"].getboolean("use_gui"),
    num_seconds=int(config["Sumo"]["num_seconds"]),
    yellow_time=int(config["Sumo"]["yellow_time"]),
    min_green=int(config["Sumo"]["min_green"]),
    max_green=int(config["Sumo"]["max_green"]),
    reward_fn=config["Sumo"]["reward_fn"],
)

step, metrics, info = 0, [], {"step": 0}
agents = {
    ts: ppo.Agent(
        env.observation_spaces[ts].shape[0],
        env.action_spaces[ts].n,
        learning_rate,
        gamma,
        gae_lambda,
        policy_clip,
        num_layers,
        width,
        model_path
    )
    for ts in env.possible_agents
}

experience_replay = replay.ReplayBuffer(buffer_size, env.observation_spaces['1'].shape[0], env.action_spaces['1'].n)

for ep in tqdm(range(num_episodes), desc="Running..", unit="epsiode"):
    info["ep"] = ep
    state, _ = env.reset()
    terminations = {a: False for a in agents}
    while not all(terminations.values()):
        reward={}
        for ts,agent in agents.items():
            action, prob, val = agent.act(state)
            next_state, rewards, terminations, truncations, infos = env.step({ts:action})
            reward[ts] = rewards[ts]
            experience_replay.store(state[ts], action[ts], prob, val, rewards[ts], terminations[ts])
            state = next_state
            
        # print(actions)
        info.update(reward)
        metrics.append(info.copy())
        if all(terminations.values()):
            for key in agents.keys():
                agents[key].learn(experience_replay.genenrate_batches(batch_size))
            break
        info["step"] += 1
        tqdm.write(f"Progress: {info['step']}/{config['Sumo'].getint('num_seconds')}", end='\r')  # Update progress in-place
    env.save_csv(output_path, ep)
    env.close()
df = pd.DataFrame(metrics)
df.to_csv(output_path + f"rewards.csv", index=False)
df['rewards'] = df[['1','2','3','4']].mean(axis=1)
df.drop(columns=['1','2','3','4'],inplace=True)
df.to_csv(output_path + f"comb_rewards.csv", index=False)
