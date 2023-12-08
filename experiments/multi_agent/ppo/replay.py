import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, ip_shape, n_actions):
        self.idx = 0
        self.ip_shape = ip_shape
        self.max_size = buffer_size
        self.state_buff = np.zeros((buffer_size,ip_shape))
        self.probs_buff = np.zeros((buffer_size,ip_shape))
        self.val_buff = np.zeros((buffer_size,ip_shape))
        self.action_buff = np.zeros((buffer_size,n_actions))
        self.reward_buff = np.zeros((buffer_size))
        self.done_buff = np.zeros((buffer_size), dtype=np.bool_)
        self.vars = [self.state_buff, self.action_buff, self.probs_buff, self.val_buff, self.reward_buff, self.done_buff]
        
    def generate_batches(self):
        n_states = len(self.ip_shape)
        batch_start = np.arange(0,n_states, self.max_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.max_size] for i in batch_start]
        
        return self.state_buff, self.action_buff, self.probs_buff, self.val_buff, self.reward_buff, self.done_buff, batches

    def store(self, experience):
        map(lambda x,y:x.append(y), list(zip(self.vars,experience)))
    
    def clear_buffer(self):
        map(lambda x: np.empty_like(x),self.vars)
        
        