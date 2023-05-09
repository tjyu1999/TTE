import random


class Memory:
    def __init__(self):
        self.router_memory = []
        self.scheduler_memory = []

    def router_push(self, state, action, reward, state_, done):
        exp = [state, action, reward, state_, done]
        self.router_memory.append(exp)

    def router_sample(self, batch_size):
        sample_indices = random.sample(range(len(self.router_memory)), batch_size)
        memory_for_train = []
        for idx in sample_indices:
            memory_for_train.append(self.router_memory[idx])

        return memory_for_train

    def scheduler_push(self, state, action, reward):
        exp = [state, action, reward]
        self.scheduler_memory.append(exp)

    def scheduler_sample(self, batch_size):
        sample_indices = random.sample(range(len(self.scheduler_memory)), batch_size)
        memory_for_train = []
        for idx in sample_indices:
            memory_for_train.append(self.scheduler_memory[idx])

        return memory_for_train

    def router_reset(self):
        self.router_memory.clear()

    def scheduler_reset(self):
        self.scheduler_memory.clear()