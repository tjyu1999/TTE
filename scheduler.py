import torch
import torch.nn as nn
import torch.optim as optim
from model import PrdModel
from memory import Memory
from param import args


class Scheduler:
    def __init__(self, device):
        self.model = PrdModel(args.edge_state_dim, args.prd_hidden_dim, device)
        self.memory = Memory(args.memory_length)
        self.device = device

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.scheduler_lr, weight_decay=args.weight_dec)

    def slot_assignment(self, state, flow_prd):
        actions = self.model(state).reshape(-1)
        valid_actions = actions[:flow_prd]
        action = torch.argmax(valid_actions).item()

        return action

    def store_pair(self, state, action, reward):
        self.memory.scheduler_push(state, action, reward)

    def sample_pair(self):
        return self.memory.scheduler_sample(args.batch_size)

    def learn(self):
        self.model.train()
        self.optimizer.zero_grad()

        preds = []
        targets = []
        sampled_memory = self.sample_pair()

        for exp in sampled_memory:
            state, action, reward = exp

            actions = self.model(state).reshape(-1)
            pred = actions[action]
            preds.append(pred)
            targets.append(reward)

        loss = self.loss_function(torch.stack(preds), torch.stack(targets))
        loss.backward()
        self.optimizer.step()

        return loss