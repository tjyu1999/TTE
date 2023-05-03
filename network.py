import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeBatchNorm(nn.Module):
    def __init__(self,
                 hidden_dim,
                 device):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        self.to(device)

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        x_bn = x_bn.squeeze()

        return x_bn


class EdgeBatchNorm(nn.Module):
    def __init__(self,
                 hidden_dim,
                 device):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        self.to(device)

    def forward(self, e):
        e = e.unsqueeze(dim=0)
        e_trans = e.transpose(1, 3).contiguous()
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()
        e_bn = e_bn.squeeze()

        return e_bn


class NodeFeat(nn.Module):

    def __init__(self,
                 hidden_dim,
                 device):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to(device)

    def forward(self, x, e_gate):
        Ux = self.U(x)
        Vx = self.V(x)
        Vx = Vx.unsqueeze(dim=0)

        gateVx = e_gate * Vx
        x_new = Ux + ((torch.sum(gateVx, dim=1)) / (1e-20 + torch.sum(e_gate, dim=1)))

        return x_new


class EdgeFeat(nn.Module):
    def __init__(self,
                 hidden_dim,
                 device):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to(device)

    def forward(self, x, e):
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(dim=0)
        Vx = Vx.unsqueeze(dim=1)
        e_new = Ue * (Vx + Wx)

        return e_new


class AGNLayer(nn.Module):
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.node_bn = NodeBatchNorm(hidden_dim, device)
        self.edge_bn = EdgeBatchNorm(hidden_dim, device)
        self.node_feat = NodeFeat(hidden_dim, device)
        self.edge_feat = EdgeFeat(hidden_dim, device)

    def forward(self, x, e):
        x_in = x
        e_in = e

        e_tmp = self.edge_bn(self.edge_feat(x_in, e_in))
        e_gate = F.softmax(e_tmp, dim=1)
        x_tmp = F.relu(self.node_bn(self.node_feat(x_in, e_gate)))

        x_next = x_in + x_tmp
        e_next = e_gate * e_in + e_in

        return x_next, e_next