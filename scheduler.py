import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
from model import PrdModel
from param import args


class Scheduler:
    def __init__(self, device):
        self.model = PrdModel(args.edge_state_dim, args.prd_hidden_dim, device)
        self.device = device

        self.memory = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.scheduler_lr, weight_decay=args.weight_dec)

    def slot_assignment(self, state, flow_prd):
        actions = self.model(state).reshape(-1)
        valid_actions = []
        for prd, val in enumerate(actions):
            if prd < flow_prd:
                valid_actions.append(val)
        valid_actions = torch.stack(valid_actions)
        distribution = Categorical(valid_actions)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.item(), log_prob

    def store(self, log_prob, reward):
        self.memory.append([log_prob, reward])

    def learn(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss_record = []
        for exp in self.memory:
            loss_record.append(-exp[0] * exp[1])

        loss = sum(loss_record) / len(loss_record)
        loss.backward()
        self.optimizer.step()

        self.memory.clear()

        return loss