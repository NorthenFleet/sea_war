import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(max_priority)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6):
        priorities = np.array(self.priorities)
        probabilities = priorities ** alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(
            len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
