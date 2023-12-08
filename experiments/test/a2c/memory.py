import numpy as np

class Memory:
    def __init__(self, buffer_size, batch_size, state_size, action_size):
        self.buffer = []
        self.max_size = buffer_size
        self.idx = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def add(self, state, action, reward, next_state):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.idx] = (state, action, reward, next_state)
            self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        for i in indices:
            print(self.buffer[i])
            state, action, reward, next_state = self.buffer[i]
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
        return self.states, self.actions, self.rewards, self.next_states