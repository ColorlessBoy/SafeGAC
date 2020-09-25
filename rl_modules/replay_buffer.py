import numpy as np
import torch

"""
the replay buffer here is basically from the openai baselines code


"""

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0, reward_scale=5.0):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        # For state normalization, and reward normalization.
        self.total_num = 0
        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.ones(obs_dim, dtype=np.float32)

        self.rew_limit = 5.0
        self.rew_mean = np.zeros(1, dtype=np.float32)
        self.rew_square_mean = np.zeros(1, dtype=np.float32)
        self.rew_std = np.ones(1, dtype=np.float32)

        self.cost_limit = 5.0
        self.cost_mean = np.zeros(1, dtype=np.float32)
        self.cost_square_mean = np.zeros(1, dtype=np.float32)
        self.cost_std = np.ones(1, dtype=np.float32)

    def store(self, obs, act, rew, cost, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost 
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        self.total_num += 1
        self.obs_mean = self.obs_mean / self.total_num * (self.total_num - 1) + np.array(obs) / self.total_num
        self.obs_square_mean = self.obs_square_mean / self.total_num * (self.total_num - 1) + np.array(obs)**2 / self.total_num
        obs_var = (self.obs_square_mean - self.obs_mean ** 2).clip(1e-4, 100)
        self.obs_std = np.sqrt(obs_var)

        self.rew_mean = self.rew_mean / self.total_num * (self.total_num - 1) + np.array(rew) / self.total_num
        self.rew_square_mean = self.rew_square_mean / self.total_num * (self.total_num - 1) + np.array(rew)**2 / self.total_num
        rew_var = (self.rew_square_mean - self.rew_mean ** 2).clip(1e-4, 100)
        self.rew_std = np.sqrt(rew_var)

        self.cost_mean = self.cost_mean / self.total_num * (self.total_num - 1) + np.array(cost) / self.total_num
        self.cost_square_mean = self.cost_square_mean / self.total_num * (self.total_num - 1) + np.array(cost)**2 / self.total_num
        cost_var = (self.cost_square_mean - self.cost_mean ** 2).clip(1e-4, 100)
        self.cost_std = np.sqrt(cost_var)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_encoder(self.rew_buf[idxs]),
                     cost=self.cost_encoder(self.cost_buf[idxs]),
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)
    
    def rew_encoder(self, rew):
        return ((np.array(rew) - self.rew_mean)/(self.rew_std + 1e-8)).clip(-self.rew_limit, self.rew_limit)

    def cost_encoder(self, cost):
        return ((np.array(cost) - self.cost_mean)/(self.cost_std + 1e-8)).clip(-self.cost_limit, self.cost_limit)