import numpy as np

class Memory:
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_size = buffer_size
        self.idx = 0

    def add(self, state, action, reward, next_state):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.idx] = (state, action, reward, next_state)
            self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in indices:
            state, action, reward, next_state = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        return states, actions, rewards, next_states