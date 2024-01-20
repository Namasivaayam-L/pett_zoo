import os, sys, configparser, time, shutil, pandas as pd
import dqn, memory
from tqdm import tqdm
import cProfile
from sumo_rl import parallel_env
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

ini_file = "experiments/multi_agent/dqn/config.ini"
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
epsilon, min_epsilon, decay = (
    config["Model"].getfloat("epsilon"),
    config["Model"].getfloat("min_epsilon"),
    config["Model"].getfloat("decay"),
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

def update_csv(info, ep):
    if not bool(info["step"]):
        mode = "w"
    else:
        mode = "a"
    for stat in ["state", "rewards"]:
        if stat == "state":
            for key in info[stat].keys():
                info[stat][key] = info[stat][key].tolist()
        info[stat]["ep"] = ep
        df = pd.DataFrame([info[stat]])
        df.to_csv(output_path + f"{stat}.csv", index=False, mode=mode, header=not bool(info["step"]))
        if stat == "rewards":
            df["rewards"] = df[["1", "2", "5", "6"]].mean(axis=1)
            df.drop(columns=["1", "2", "5", "6"], inplace=True)
            df.to_csv(output_path + f"comb_rewards.csv", index=False, mode=mode, header=not bool(info["step"]))
        del info[stat]["ep"]


info = {"step": 0, "state": {}, "rewards": {}}
agents = {
    ts: dqn.DQN(
        ts,
        env.action_spaces[ts].n,
        env.observation_spaces[ts].shape[0],
        width,
        num_layers,
        batch_size,
        gamma,
        learning_rate,
        model_path,
        fine_tune_model_path,
    )
    for ts in env.possible_agents
}
# experience_replay = memory.Memory(buffer_size,env.observation_spaces['1'].shape[0], env.action_spaces['1'].n)
info = {"step": 0, "state":{}, "rewards":{}}
experience_replay = memory.Memory(buffer_size)
for ep in tqdm(range(num_episodes), desc="Running..", unit="epsiode"):
    state, _ = env.reset()
    terminations = {a: False for a in agents}
    # Create a cProfile object
    profiler = cProfile.Profile()
    # Start profiling
    profiler.enable()
    while not all(terminations.values()):
        actions = {ts: agents[ts].act(state[ts], epsilon) for ts in env.possible_agents}
        next_state, rewards, terminations, truncations, infos = env.step(actions)
        info["state"].update(state)
        info["rewards"].update(rewards)
        if all(terminations.values()):
            for key in agents.keys():
                agents[key].learn(experience_replay.sample(batch_size))
            break
        for ts in env.possible_agents:
            experience_replay.add(state[ts], actions[ts], rewards[ts], next_state[ts])
        state = next_state
        update_csv(info, ep)
        info["step"] += 1
        tqdm.write(f"Progress: {info['step']}/{config['Sumo'].getint('num_seconds')}", end="\r")
    profiler.disable()
    profiling_output_path = f"experiments/multi_agent/dqn/profiling/{ep}.prof"
    profiler.dump_stats(profiling_output_path)
    epsilon = max(epsilon * decay, min_epsilon)
    env.save_csv(output_path, ep)
    env.close()
