import numpy as np

class Memory:
    def __init__(self, buffer_size, ip_shape, n_actions):
        self.idx = 0
        self.max_size = buffer_size
        self.state_buff = np.zeros((buffer_size,ip_shape))
        self.action_buff = np.zeros((buffer_size,n_actions))
        self.reward_buff = np.zeros((buffer_size))
        self.nxt_state_buff = np.zeros((buffer_size,ip_shape))
        self.done_buff = np.zeros((buffer_size), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        idx = self.idx % self.max_size
        self.state_buff[idx] = state
        self.action_buff[idx] = action
        self.reward_buff[idx] = reward
        self.nxt_state_buff[idx] = next_state
        self.done_buff[idx] = done
        self.idx += 1

    def sample(self, batch_size):
        max_mem = min(self.idx, self.max_size)
        batch = np.random.choice(max_mem, batch_size)
        return self.state_buff[batch],self.action_buff[batch],self.reward_buff[batch],self.nxt_state_buff[batch],self.done_buff[batch]