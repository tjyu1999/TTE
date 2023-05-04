import torch
import torch.nn as nn
import torch.optim as optim
from model import AGNModel
from memory import Memory
from param import args


class Router:
    def __init__(self, adj_mat, dist_mat, neighbors, device):
        self.model = AGNModel(args.agn_hidden_dim, args.layer_num, device)
        self.memory = Memory()
        self.adj_mat = adj_mat
        self.dist_mat = dist_mat
        self.neighbors = neighbors
        self.device = device
        
        self.node_num = len(adj_mat)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.router_lr, weight_decay=args.weight_dec)

    def select_action(self, curr_node_idx, node_state, adj_mat, dist_mat):
        if self.adj_mat is None or self.dist_mat is None:
            self.adj_mat = adj_mat
            self.dist_mat = dist_mat

        actions = self.model(node_state, adj_mat, dist_mat)
        actions = actions.reshape(self.node_num, self.node_num)[curr_node_idx]

        valid_actions = []
        mask = self.neighbors[curr_node_idx]
        for idx in mask:
            valid_actions.append(actions[idx].item())
        next_node_idx = mask[valid_actions.index(max(valid_actions))]

        return next_node_idx

    @torch.no_grad()
    def select_action_for_val(self):
        pass

    def store_pair(self, state, action, reward, state_, done):
        self.memory.push(state, action, reward, state_, done)

    def sample_pair(self):
        return self.memory.sample(args.batch_size)

    def learn(self):
        self.model.train()
        self.optimizer.zero_grad()

        preds = []
        targets = []
        sampled_memory = self.sample_pair()

        adj_mat = torch.from_numpy(self.adj_mat).float().to(self.device)
        dist_mat = torch.from_numpy(self.dist_mat).float().to(self.device)
        for exp in sampled_memory:
            state, action, reward, state_, done = exp

            actions = self.model(state.to(self.device), adj_mat, dist_mat).reshape(self.node_num, self.node_num)
            pred = actions[action[0], action[1]]
            next_actions = self.model(state_.to(self.device), adj_mat, dist_mat)[action[1]].detach()
            target = reward.to(self.device) + args.gamma * torch.max(next_actions) * (1 - done)

            preds.append(pred)
            targets.append(target)

        loss = self.loss_function(torch.stack(preds), torch.stack(targets))
        loss.backward()
        self.optimizer.step()

        self.memory.clear()
        preds.clear()
        targets.clear()

        return loss
