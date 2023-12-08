import numpy as np

class Memory:
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_size = buffer_size
        self.idx = 0

    def store(self, state, action, reward, next_state):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.idx] = (state, action, reward, next_state)
            self.idx = (self.idx + 1) % self.max_size

    def sample_batch(self, batch_size, ts):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in indices:
            state, action, reward, next_state = self.buffer[i]
            return state, action, reward, next_state
        #     states.append(state[ts])
        #     actions.append(action)
        #     rewards.append(reward)
        #     next_states.append(next_state[ts])
        # return states, actions, rewards, next_states