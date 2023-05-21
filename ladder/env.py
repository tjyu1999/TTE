import numpy as np
from graph import Graph
from param import args


class Env:
    def __init__(self, data):
        self.data = data
        self.graph = Graph(data)

        self.adj_mat = self.graph.adj_mat
        self.dis_mat = self.graph.dis_mat
        self.neighbors = self.graph.neighbors
        self.n2e = self.graph.n2e

        self.node_num = len(self.adj_mat)
        self.x_dim = args.x_dim
        self.e_dim = args.e_dim
        self.p_dim = args.p_dim
        self.hyper_prd = args.hyper_prd
        self.slot_num = args.slot_num

        self.flow_info = []
        self.visited_node = []
        self.visited_edge = []
        self.deadline = 0

    def load_info(self, i):
        self.flow_info = self.data.flow_info[i]
        self.visited_node.append(self.flow_info[0])
        self.deadline = self.flow_info[4]

    def r_state(self):
        x = np.zeros([self.node_num, self.x_dim])
        e = np.zeros([self.node_num, self.node_num, self.e_dim])

        for node in self.graph.nodes.values():
            x[node.idx][0] = node.buff_status / node.buff_size
            x[node.idx][1] = 1 if node.idx == self.visited_node[-1] else 0
            x[node.idx][2] = 1 if node.idx == self.flow_info[1] else 0
            x[node.idx][3] = 1 if node.idx in self.neighbors[self.visited_node[-1]] \
                                  and not self.check_cycle(node.idx) else 0
        for i in range(self.node_num):
            for j in self.neighbors[i]:
                e_i = self.n2e.index([i, j])
                e[i][j][0] = sum(self.graph.edges[e_i].slot_status) / len(self.graph.edges[e_i].slot_status)

        return [x, e]

    def s_state(self):
        p = np.zeros([self.hyper_prd, self.p_dim])

        for prd in range(self.hyper_prd):
            valid = 1
            if prd < self.flow_info[2]:
                pos = self.graph.edges[self.visited_edge[-1]].find_slot(prd, self.flow_info[2])
                if pos:
                    dly = pos[0]
                    if dly < self.deadline:
                        valid = dly / self.deadline
            p[prd][0] = valid
            p[prd][1] = sum(self.graph.edges[self.visited_edge[-1]].slot_status[
                        prd * self.slot_num:(prd + 1) * self.slot_num]) / self.slot_num

        return p

    def r_step(self, j):
        i = self.visited_node[-1]
        e_i = self.n2e.index([i, j])

        done = -1
        reward = -10
        if j not in self.visited_node and not self.check_cycle(j):
            self.visited_node.append(j)
            self.visited_edge.append(e_i)
            if j == self.flow_info[1]:
                done = 1
                reward = 10
            elif j != self.flow_info[1]:
                done = 0
                reward = 0

        return done, reward, self.r_state()

    def s_step(self, prd):
        e_i = self.visited_edge[-1]
        pos = self.graph.edges[e_i].find_slot(prd, self.flow_info[2])

        done = -1
        reward = -10
        if pos:
            dly = pos[0]
            if self.deadline > dly:
                done = 1
                reward = 10 - (pos[0] / self.deadline)
                if len(self.visited_edge) > 1:
                    self.deadline -= dly

        return done, reward, pos

    def check_cycle(self, node_idx):
        visited = self.visited_node.copy()
        dst = self.flow_info[1]

        if node_idx == dst:
            return False
        return self.depth_first_search(node_idx, visited)

    def depth_first_search(self, node_idx, visited):
        visited.append(node_idx)
        dst = self.flow_info[1]

        for adj_idx in self.neighbors[node_idx]:
            if adj_idx == dst:
                return False
            elif adj_idx in visited:
                continue
            elif adj_idx not in visited:
                return self.depth_first_search(adj_idx, visited)

        return True

    def train_schedule(self, actions):
        for act in actions:
            e_i, pos = act
            self.graph.edges[e_i].e_node.occupy_buff(self.flow_info[3])
            self.graph.edges[e_i].occupy_slot(pos)

    def val_schedule(self, e_i, pos):
        self.graph.edges[e_i].e_node.occupy_buff(self.flow_info[3])
        self.graph.edges[e_i].occupy_slot(pos)

    def failure_compensate(self, table, e_i, e_j):
        redo = []
        for f_i in range(len(table)):
            if f'{e_i}' in table[f_i].keys() or f'{e_j}' in table[f_i].keys():
                redo.append(f_i)

        for f_i in redo:
            for e_k in table[f_i]:
                e = self.graph.edges[e_k]
                self.graph.nodes[e.e_node.idx].compensate_buff(self.data.flow_info[f_i][3])
                e.compensate_slot(table[f_i][e_k])

        return redo

    def renew(self):
        self.visited_node.clear()
        self.visited_edge.clear()

    def reset(self):
        self.renew()
        for node in self.graph.nodes.values():
            node.reset()
        for edge in self.graph.edges.values():
            edge.reset()