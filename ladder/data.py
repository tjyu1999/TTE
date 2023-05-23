import random
import numpy as np
from param import args


class Ladder:
    def __init__(self, node_num, flow_num):
        self.node_num = node_num
        self.flow_num = flow_num

        self.buff_size = args.buff_size
        self.flow_prd = args.flow_prd
        self.flow_len = args.flow_len
        self.flow_dly = args.flow_dly

        self.flow_info = {}
        self.adj_mat = []

    def gen_info(self):
        for i in range(self.flow_num):
            src = random.randint(0, self.node_num - 1)
            dst = random.randint(0, self.node_num - 1)
            while src == dst:
                dst = random.randint(0, self.node_num - 1)
            prd = self.flow_prd[random.randint(0, len(self.flow_prd) - 1)]
            length = random.randint(self.flow_len[0], self.flow_len[1])
            dly = random.randint(self.flow_dly[0], self.flow_dly[1])

            self.flow_info[i] = [src, dst, prd, length, dly]

    def gen_mat(self):
        self.adj_mat = np.zeros([self.node_num, self.node_num])
        edges = []

        i = 0
        while i + 1 < self.node_num:
            edges.append([i, i + 1])
            i += 2
        i = 0
        while i + 2 < self.node_num:
            edges.append([i, i + 2])
            i += 1

        for edge in edges:
            self.adj_mat[edge[0]][edge[1]] = 1
            self.adj_mat[edge[1]][edge[0]] = 1

    def overall(self):
        self.gen_info()
        self.gen_mat()