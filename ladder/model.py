import torch
import torch.nn as nn
import torch.nn.functional as F
from network import AGNLayer
from param import args


class AGNModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.x_layer = nn.Linear(args.x_dim, args.agn_hidden_dim, bias=False)
        self.e_layer = nn.Linear(args.e_dim, args.agn_hidden_dim // 3, bias=False)
        self.a_layer = nn.Linear(1, args.agn_hidden_dim // 2, bias=False)
        self.d_layer = nn.Linear(1, args.agn_hidden_dim // 2, bias=False)
        self.fc_layer = nn.Linear(args.agn_hidden_dim, 1)
        self.agn_layers = nn.ModuleList(AGNLayer(args.agn_hidden_dim, device)
                                        for _ in range(args.layer_num))

        self.state_path = 'models/agn_state.pt'
        self.to(device)

    def forward(self, x, e, a, d):
        x_ = self.x_layer(x)
        # e_ = self.e_layer(e)
        a_ = self.a_layer(a.unsqueeze(dim=2))
        d_ = self.d_layer(d.unsqueeze(dim=2))
        e_ = torch.cat((a_, d_), dim=2)
        for layer in self.agn_layers:
            x_, e_ = layer(x_, e_)
        out = self.fc_layer(e_)

        return out

    def save_state(self):
        torch.save(self.state_dict(), self.state_path)

    def load_state(self):
        self.load_state_dict(torch.load(self.state_path))


class PrdModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fc_layer_1 = nn.Linear(args.p_dim, args.prd_hidden_dim[0])
        self.fc_layer_2 = nn.Linear(args.prd_hidden_dim[0], args.prd_hidden_dim[1])
        self.fc_layer_3 = nn.Linear(args.prd_hidden_dim[1], args.prd_hidden_dim[2])
        self.fc_layer_4 = nn.Linear(args.prd_hidden_dim[2], 1)

        self.state_path = 'models/prd_state.pt'
        self.to(device)

    def forward(self, p):
        p = F.elu(self.fc_layer_1(p))
        p = F.elu(self.fc_layer_2(p))
        p = F.elu(self.fc_layer_3(p))
        out = self.fc_layer_4(p)

        return out

    def save_state(self):
        torch.save(self.state_dict(), self.state_path)

    def load_state(self):
        self.load_state_dict(torch.load(self.state_path))