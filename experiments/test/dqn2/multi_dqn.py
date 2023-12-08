import dqn
import memory
class MultiAgentDQN:
    def __init__(self, ts_ids, num_actions, num_states):
        self.ts_ids = ts_ids
        self.batch_size = 500
        self.num_actions = num_actions
        self.num_states = num_states
        self.agents = {ts: dqn.DQN(num_actions, num_states.shape[0], self.batch_size) for ts in self.ts_ids}
        self.experience_replay = memory.Memory(10000)

    def train(self, env, num_episodes):
        for ep in range(num_episodes):
            print('episode: ',ep)
            state = env.reset()
            eps = 0.7
            done = {"__all__": False}
            while not done["__all__"]:
                # print(f'Ep: {ep} not done')
                actions = { ts: self.agents[ts].act(state[ts], self.experience_replay, ts, eps) for ts in self.agents.keys()} 
                next_state, rewards, done, _ = env.step(action=actions)
                self.experience_replay.store(state, actions, rewards, next_state)
                state = next_state
                if done["__all__"]:
                    print(f'Ep: {ep} done, Agent learning')
                    for ts in self.agents.keys():
                        self.agents[ts].learn(self.experience_replay.sample_batch(self.batch_size,ts))
            eps -= 0.09
            env.save_csv('outputs/multi_agent/dqn/', ep)
            env.close()
    # def evaluate(self, env, num_episodes):
    #     total_rewards = []
    #     for _ in range(num_episodes):
    #         state = env.reset()
    #         done = False
    #         episode_reward = 0
    #         while not done:
    #             actions = self.select_actions(state, epsilon=0.0)
    #             next_observations, rewards, done, _ = env.step(actions)

    #             episode_reward += rewards
    #             state = next_observations

    #         total_rewards.append(episode_reward)
    #     return np.mean(total_rewards)