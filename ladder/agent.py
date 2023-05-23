import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import AGNModel, PrdModel
from param import args


class Agent:
    def __init__(self, adj_mat, dis_mat, neighbors, device):
        self.adj_mat = adj_mat
        self.dis_mat = dis_mat
        self.neighbors = neighbors
        self.device = device

        self.agn_model = AGNModel(device)
        self.prd_model = PrdModel(device)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam([{'params': self.agn_model.parameters(), 'lr': args.r_lr},
                                     {'params': self.prd_model.parameters(), 'lr': args.s_lr}],
                                    weight_decay=args.weight_dec)

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.r_memory = []
        self.s_memory = []

    def train_model(self):
        self.agn_model.train()
        self.prd_model.train()

    def eval_model(self):
        self.agn_model.eval()
        self.prd_model.eval()

    def save_model(self):
        self.agn_model.save_state()
        self.prd_model.save_state()

    def load_model(self):
        self.agn_model.load_state()
        self.prd_model.load_state()

    def r_make_action(self, i, x, e):
        x = x.to(self.device)
        e = e.to(self.device)
        a_mat = torch.from_numpy(self.adj_mat).float().to(self.device)
        d_mat = torch.from_numpy(self.dis_mat).float().to(self.device)
        actions = self.agn_model(x, e, a_mat, d_mat)
        actions = actions.squeeze()[i]

        valid_actions = []
        mask = self.neighbors[i]
        for n in mask:
            valid_actions.append(actions[n].item())
        action = mask[valid_actions.index(max(valid_actions))]

        return action

    def s_make_action(self, s, flow_prd):
        s = s.to(self.device)
        actions = self.prd_model(s).reshape(-1)
        actions = actions[:flow_prd]
        action = torch.argmax(actions).item()

        return action

    def r_make_action_val(self, i, x, e, mask):
        x = x.to(self.device)
        e = e.to(self.device)
        a_mat = torch.from_numpy(self.adj_mat).float().to(self.device)
        d_mat = torch.from_numpy(self.dis_mat).float().to(self.device)
        actions = self.agn_model(x, e, a_mat, d_mat)
        actions = actions.squeeze()[i]

        valid_actions = []
        for n in mask:
            valid_actions.append(actions[n].item())
        action = mask[valid_actions.index(max(valid_actions))]

        return action

    def s_make_action_val(self, s, mask):
        s = s.to(self.device)
        actions = self.prd_model(s).reshape(-1)

        valid_actions = []
        for i in mask:
            valid_actions.append(actions[i].item())
        action = mask[valid_actions.index(max(valid_actions))]

        return action

    def r_push(self, s, a, r, s_, d):
        self.r_memory.append([s, a, r, s_, d])

    def s_push(self, s, a, r):
        self.s_memory.append([s, a, r])

    def sample_memory(self):
        r_sample = []
        s_sample = []
        for i, j in zip(random.sample(range(len(self.r_memory)), self.batch_size),
                        random.sample(range(len(self.s_memory)), self.batch_size)):
            r_sample.append(self.r_memory[i])
            s_sample.append(self.s_memory[j])

        return r_sample, s_sample

    def clear_memory(self):
        self.r_memory.clear()
        self.s_memory.clear()

    def learn(self):
        self.optimizer.zero_grad()
        a_mat = torch.from_numpy(self.adj_mat).float().to(self.device)
        d_mat = torch.from_numpy(self.dis_mat).float().to(self.device)

        r_pair = [[], []]
        s_pair = [[], []]
        r_sample, s_sample = self.sample_memory()

        for r_exp, s_exp in zip(r_sample, s_sample):
            s, a, r, s_, d = r_exp
            r_actions = self.agn_model(s[0].to(self.device), s[1].to(self.device), a_mat, d_mat).squeeze()
            r_actions_ = self.agn_model(s_[0].to(self.device), s_[1].to(self.device), a_mat, d_mat)[a[1]].detach()
            r_pred = r_actions[a[0]][a[1]]
            r_target = r.to(self.device) + self.gamma * torch.max(r_actions_) * (1 - d)
            r_pair[0].append(r_pred)
            r_pair[1].append(r_target)

            _s, _a, _r = s_exp
            s_actions = self.prd_model(_s.to(self.device)).reshape(-1)
            s_pred = s_actions[_a]
            s_target = _r.to(self.device)
            s_pair[0].append(s_pred)
            s_pair[1].append(s_target)

        r_loss = self.loss_function(torch.stack(r_pair[0]), torch.stack(r_pair[1]))
        s_loss = self.loss_function(torch.stack(s_pair[0]), torch.stack(s_pair[1]))
        loss = r_loss + s_loss
        loss.backward()
        self.optimizer.step()

        self.save_model()
        self.clear_memory()

        return r_loss, s_loss