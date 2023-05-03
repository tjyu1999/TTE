import torch
import torch.nn as nn
import torch.optim as optim
from model import PrdModel
from param import args


class Scheduler:
    def __init__(self, device):
        self.model = PrdModel(args.edge_state_dim, args.prd_hidden_dim, device)
        self.device = device

        self.preds = []
        self.targets = []

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.scheduler_lr, weight_decay=args.weight_dec)

    def slot_assignment(self, state, flow_prd):
        actions = self.model(state).reshape(-1)
        valid_actions = actions[:flow_prd]

        action = torch.argmax(valid_actions).item()
        value = valid_actions[action]

        return action, value

    def store_pair(self, value, reward):
        self.preds.append(value)
        self.targets.append(reward)

    def learn(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.loss_function(torch.stack(self.preds), torch.stack(self.targets))
        loss.backward()
        self.optimizer.step()

        self.preds.clear()
        self.targets.clear()

        return loss