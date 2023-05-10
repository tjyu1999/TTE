import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=4171)
parser.add_argument('--hyper_prd', type=int, default=256)
parser.add_argument('--slot_num', type=int, default=64)
parser.add_argument('--node_num', type=int, default=12)
parser.add_argument('--buff_size', type=int, default=[1e6, 1e6])
parser.add_argument('--flow_prd', type=int, default=[4, 8, 16, 32, 64, 128, 256])
parser.add_argument('--flow_length', type=int, default=[64, 1518])
parser.add_argument('--flow_dly', type=int, default=[16384, 32768])
parser.add_argument('--node_state_dim', type=int, default=4)
parser.add_argument('--edge_state_dim', type=int, default=3)
parser.add_argument('--agn_hidden_dim', type=int, default=300)
parser.add_argument('--layer_num', type=int, default=20)
parser.add_argument('--prd_hidden_dim', type=int, default=[16, 4])
parser.add_argument('--router_lr', type=int, default=1e-5)
parser.add_argument('--scheduler_lr', type=int, default=1e-3)
parser.add_argument('--weight_dec', type=int, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gamma', type=int, default=0.9)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--train_flow_num', type=int, default=1000)
parser.add_argument('--val_times', type=int, default=100)
parser.add_argument('--val_flow_num', type=int, default=10000)

args = parser.parse_args()