import random
import numpy as np
from param import args


class Node:
    def __init__(self, idx, buff_size):
        self.idx = idx
        self.buff_size = buff_size
        self.buff_status = buff_size

    def occupy_buff(self, flow_len):
        self.buff_status -= flow_len

    def compensate_buff(self, flow_len):
        self.buff_status += flow_len

    def reset(self):
        self.buff_status = self.buff_size


class Edge:
    def __init__(self, idx, s_node, e_node):
        self.idx = idx
        self.s_node = s_node
        self.e_node = e_node

        self.hyper_prd = args.hyper_prd
        self.slot_num = args.slot_num
        self.slot_status = [1 for _ in range(self.hyper_prd * self.slot_num)]

    def find_slot(self, prd, flow_prd):
        frames = int(self.hyper_prd / flow_prd)

        for slot_idx in range(prd * self.slot_num, (prd + 1) * self.slot_num):
            if self.slot_status[slot_idx] != 1:
                continue
            pos_of_slot = []
            failed = False

            for frame in range(frames):
                temp_slot_idx = slot_idx + frame * flow_prd * self.slot_num
                if self.slot_status[temp_slot_idx] == 1:
                    pos_of_slot.append(temp_slot_idx)
                    continue
                else:
                    failed = True
                    break

            if not failed:
                return pos_of_slot
            elif failed:
                continue

        return []

    def occupy_slot(self, pos):
        for p in pos:
            self.slot_status[p] -= 1

    def compensate_slot(self, pos):
        for p in pos:
            self.slot_status[p] += 1

    def reset(self):
        self.slot_status = [1 for _ in range(self.hyper_prd * self.slot_num)]


class Graph:
    def __init__(self, data):
        self.data = data
        self.node_num = data.node_num
        self.buff_size = args.buff_size

        self.adj_mat = []
        self.dis_mat = []
        self.neighbors = {}
        self.n2e = []
        self.nodes = {}
        self.edges = {}

        self.load_mat()
        self.load_info()

    def load_mat(self):
        self.adj_mat = self.data.adj_mat
        self.dis_mat = np.copy(self.adj_mat)

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j and self.adj_mat[i][j] == 0:
                    self.dis_mat[i][j] = float('inf')
        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if self.dis_mat[i][j] > self.dis_mat[i][k] + self.dis_mat[k][j]:
                        self.dis_mat[i][j] = self.dis_mat[i][k] + self.dis_mat[k][j]

        for i in range(self.node_num):
            self.neighbors[i] = []
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.adj_mat[i][j] == 1:
                    self.neighbors[i].append(j)
                    self.n2e.append([i, j])

    def load_info(self):
        for i in range(self.node_num):
            self.nodes[i] = Node(int(i), self.buff_size)

        idx = 0
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.adj_mat[i][j] == 0:
                    continue
                self.edges[idx] = Edge(idx, self.nodes[i], self.nodes[j])
                idx += 1

    def failure(self):
        e_i = random.randint(0, len(self.edges) - 1)
        s_node_i = self.edges[e_i].s_node.idx
        e_node_i = self.edges[e_i].e_node.idx
        e_j = self.n2e.index([e_node_i, s_node_i])

        self.adj_mat[s_node_i][e_node_i] = 0
        self.adj_mat[e_node_i][s_node_i] = 0
        self.edges.pop(e_i)
        self.edges.pop(e_j)

        self.load_mat()

        return e_i, e_j