import random
import numpy as np
from param import args


class Ladder:
    def __init__(self, flow_num=100):
        self.node_num = args.node_num
        self.flow_num = flow_num

        self.adj_mat = []
        self.node_info = {}
        self.flow_info = {}

    def gen_adj_mat(self):
        self.adj_mat = np.zeros((self.node_num, self.node_num))
        edges = []

        idx = 0
        while idx + 1 < self.node_num:
            edges.append([idx, idx + 1])
            idx += 2
        idx = 0
        while idx + 2 < self.node_num:
            edges.append([idx, idx + 2])
            idx += 1

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    self.adj_mat[i, j] = 0
        for edge in edges:
            self.adj_mat[edge[0], edge[1]] = 1
            self.adj_mat[edge[1], edge[0]] = 1

    def gen_node_info(self):
        for idx in range(self.node_num):
            self.node_info[idx] = random.randint(args.buff_size[0], args.buff_size[1])

    def gen_flow_info(self):
        for idx in range(self.flow_num):
            src = random.randint(0, self.node_num - 1)
            dst = random.randint(0, self.node_num - 1)
            while src == dst:
                dst = random.randint(0, self.node_num - 1)
            prd = args.flow_prd[random.randint(0, len(args.flow_prd) - 1)]
            length = random.randint(args.flow_length[0], args.flow_length[1])
            dly = random.randint(args.flow_dly[0], args.flow_dly[1])

            self.flow_info[idx] = [src, dst, prd, length, dly]

    def gen_all_data(self):
        self.adj_mat = []
        self.node_info = {}
        self.flow_info = {}

        self.gen_adj_mat()
        self.gen_node_info()
        self.gen_flow_info()