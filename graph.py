import numpy as np
from param import args


class Node:
    def __init__(self, idx, buff_size):
        self.idx = idx
        self.buff_size = buff_size
        self.buff_status = buff_size

    def reset(self):
        self.buff_status = self.buff_size


class Edge:
    def __init__(self, idx, start_node, end_node):
        self.idx = idx
        self.start_node = start_node
        self.end_node = end_node

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

    def occupy_slot(self, pos_of_slot):
        for pos in pos_of_slot:
            self.slot_status[pos] -= 1

    def check_slot(self, pos_of_slot):
        check = True
        for pos in pos_of_slot:
            if self.slot_status[pos] != 1:
                check = False
                break

        return check

    def reset(self):
        self.slot_status = [1 for _ in range(self.hyper_prd * self.slot_num)]


class Graph:
    def __init__(self, data):
        self.data = data
        self.node_num = self.data.node_num

        self.nodes = {}
        self.edges = {}
        self.adj_mat = []
        self.dist_mat = []

        self.neighbors = {}
        self.node_to_edge = []

        self.get_and_load_all()

    def load_nodes(self):
        for idx, (_, buff_size) in zip(range(len(self.data.node_info)), self.data.node_info.items()):
            self.nodes[idx] = Node(int(idx), buff_size)

    def load_adj_mat(self):
        self.adj_mat = self.data.adj_mat

    def get_edges(self):
        start_from_node = {}

        for idx in range(self.node_num):
            start_from_node[idx] = []
        idx = 0
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.adj_mat[i][j] == 0 or i == j:
                    continue
                self.edges[idx] = Edge(idx, self.nodes[i], self.nodes[j])
                start_from_node[i].append(idx)
                idx += 1

    def get_dist_mat(self):
        self.dist_mat = np.copy(self.adj_mat)

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    self.dist_mat[i][j] = 0
                elif self.adj_mat[i][j] != 0:
                    self.dist_mat[i][j] = self.adj_mat[i][j]
                else:
                    self.dist_mat[i][j] = float('inf')

        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if self.dist_mat[i][j] > self.dist_mat[i][k] + self.dist_mat[k][j]:
                        self.dist_mat[i][j] = self.dist_mat[i][k] + self.dist_mat[k][j]

        return self.dist_mat

    def get_and_load_all(self):
        self.load_nodes()
        self.load_adj_mat()
        self.get_edges()
        self.get_dist_mat()

        for i in range(self.node_num):
            self.neighbors[i] = []
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.adj_mat[i][j] == 1:
                    self.neighbors[i].append(j)
                    self.node_to_edge.append([i, j])