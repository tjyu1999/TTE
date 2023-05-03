import random


class Memory:
    def __init__(self):
        self.replay_buffer = []
        self.memory_for_train = []

    def push(self, state, action, reward, state_, done):
        exp = [state, action, reward, state_, done]
        self.replay_buffer.append(exp)

    def sample(self, batch_size):
        sample_indices = random.sample(range(len(self.replay_buffer)), batch_size)
        for idx in sample_indices:
            self.memory_for_train.append(self.replay_buffer[idx])

        return self.memory_for_train

    def clear(self):
        self.replay_buffer.clear()
        self.memory_for_train.clear()