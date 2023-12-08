import os, sys, configparser, time, shutil,pandas as pd
import memory
from sumo_rl import SumoEnvironment
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import ReplayBuffer

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

time = time.strftime("%H:%M:%S_%d-%m-%Y")

ini_file = "experiments/multi_agent/a2c/config.ini"
config = configparser.ConfigParser()
config.read(ini_file)

output_path = config["Sumo"]["out_dir"]
model_path = output_path + "model/model.keras"
csv_path = output_path + config["Model"]["Name"]
log_path = output_path + "logs/"
fine_tune_actor_path, fine_tune_critic_path = None, None
if config["Model"].getboolean("fine_tune"):
    fine_tune_actor_path, fine_tune_critic_path = (
        config["Model"]["fine_tune_actor_path"],
        config["Model"]["fine_tune_critic_path"],
    )

batch_size, gamma, learning_rate, tau = (
    config["Model"].getint("batch_size"),
    config["Model"].getfloat("gamma"),
    config["Model"].getfloat("learning_rate"),
    config["Model"].getfloat("tau"),
)
num_episodes, buffer_size = config["Model"].getint("num_episodes"), config["Memory"].getint("buffer_size")
num_layers, width = config["Model"].getint("num_layers"), config["Model"].getint("width")

os.makedirs(output_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(output_path + "model/", exist_ok=True)
shutil.copyfile(ini_file, output_path + "config.ini")
from stable_baselines3.common.logger import configure

# Configure and set the logger for your A2C agent
logger = configure(folder="a2c_logs", format_strings=["stdout", "tensorboard"])

env = SumoEnvironment(
    net_file=config["Sumo"]["net_file"],
    route_file=config["Sumo"]["route_file"],
    out_csv_name=csv_path,
    use_gui=config["Sumo"].getboolean("use_gui"),
    num_seconds=int(config["Sumo"]["num_seconds"]),
    yellow_time=int(config["Sumo"]["yellow_time"]),
    min_green=int(config["Sumo"]["min_green"]),
    max_green=int(config["Sumo"]["max_green"]),
)

agents = {
    ts: A2C(
        env=env,
        policy=config["Model"]["policy"],
        learning_rate=float(config["Model"]["learning_rate"]),
        gamma=float(config["Model"]["gamma"]),
        n_steps=int(config["Model"]["n_steps"]),
        use_rms_prop=config["Model"].getboolean("use_rms_prop"),
        device=config["Model"]["device"],
        tensorboard_log=config["Model"]["tensorboard_log"],
        verbose=int(config["Model"]["verbose"]),
    )
    for ts in env.ts_ids
}
for ts in env.ts_ids:
    agents[ts].set_logger(logger=logger)
no_of_train_iters =config['Model'].getint('no_of_train_iters')
step,metrics,info = 0,[],{"step":0}
experience_replay = memory.Memory(buffer_size, batch_size, env.observation_space.shape[0], env.action_space.n)
for ep in range(num_episodes):
    info['ep'] = ep
    state = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        print("Before predicting actions")
        actions = {ts: agents[ts].predict(state[ts])[0] for ts in env.ts_ids}
        print("Actions predicted:", actions)

        next_state, rewards, done, _ = env.step(actions)
        print("Environment stepped. Done:", done)

        info.update(rewards)
        print("Info updated with rewards:", info)

        metrics.append(info.copy())
        print("Metrics updated with a copy of info")

        if done['__all__']:
            print("All terminations reached. Training agents.")

            for key in agents.keys():
                # for _ in range(no_of_train_iters):
                    # state,actions,rewards,next_state = agents[key].collect_rollouts(env,batch_size)
                agents[key].learn(total_timesteps=int(config["Train"]["total_timesteps"]), progress_bar=config["Train"].getboolean('progress_bar'))
                    # agents[key].rollout_buffer.sa
                # print(f"Agent {key} trained for {no_of_train_iters} iterations.")

            break

        # for ts in env.ts_ids:
        #     agents[ts].rollout_buffer.add(state[ts], actions[ts], rewards[ts], next_state[ts],)
        print("Experience replay updated.")

        state = next_state
        info['step'] += 1
        print("State and step updated.")

    print("Training loop completed.")

    env.save_csv(output_path, ep)
    env.close()
df = pd.DataFrame(metrics)
df.to_csv(output_path+f'rewards.csv',index=False)