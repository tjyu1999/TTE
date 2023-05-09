import torch
import torch.nn as nn
import torch.nn.functional as F
from network import AGNLayer


class AGNModel(nn.Module):
    def __init__(self, hidden_dim, layer_num, device):
        super().__init__()
        self.node_layer = nn.Linear(4, hidden_dim, bias=False)
        self.adj_layer = nn.Linear(1, hidden_dim // 2, bias=False)
        self.dist_layer = nn.Linear(1, hidden_dim // 2, bias=False)
        self.fc_layer = nn.Linear(hidden_dim, 1)
        self.agn_layers = nn.ModuleList(AGNLayer(hidden_dim, device) for _ in range(layer_num))
        self.to(device)

        self.state_path = 'models/agn_state.pt'
        self.best_state_path = 'models/best_agn_state.pt'

    def forward(self, x, adj_mat, dist_mat):
        x = self.node_layer(x)
        adj_mat = self.adj_layer(adj_mat.unsqueeze(dim=2))
        dist_mat = self.dist_layer(dist_mat.unsqueeze(dim=2))
        e = torch.cat((adj_mat, dist_mat), dim=2)

        for agn_layer in self.agn_layers:
            x, e = agn_layer(x, e)

        out = self.fc_layer(e)

        return out

    def save_state(self, is_best):
        torch.save(self.state_dict(), self.state_path)
        if is_best:
            torch.save(self.state_dict(), self.best_state_path)

    def load_state(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_state_path))
        else:
            self.load_state_dict(torch.load(self.state_path))


class PrdModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.fc_layer_1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc_layer_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc_layer_3 = nn.Linear(hidden_dim[1], 1)
        self.to(device)

        self.state_path = 'models/prd_state.pt'
        self.best_state_path = 'models/best_prd_state.pt'

    def forward(self, e):
        e = F.leaky_relu(self.fc_layer_1(e))
        e = F.leaky_relu(self.fc_layer_2(e))
        out = self.fc_layer_3(e)

        return out

    def save_state(self, is_best):
        torch.save(self.state_dict(), self.state_path)
        if is_best:
            torch.save(self.state_dict(), self.best_state_path)

    def load_state(self, is_best):
        if is_best:
            self.load_state_dict(torch.load(self.best_state_path))
        else:
            self.load_state_dict(torch.load(self.state_path))