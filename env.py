import numpy as np
from graph import Graph
from param import args


class Env:
    def __init__(self, data):
        self.data = data
        self.graph = Graph(data)

        self.adj_mat = self.graph.adj_mat
        self.dist_mat = self.graph.dist_mat
        self.neighbors = self.graph.neighbors
        self.node_to_edge = self.graph.node_to_edge

        self.flow_info = []
        self.visited_node = []
        self.visited_edge = []
        self.actions = {}
        self.deadline = 0

        self.hyper_prd = args.hyper_prd
        self.slot_num = args.slot_num

    def get_info(self, flow_idx):
        self.flow_info = self.data.flow_info[flow_idx]
        self.visited_node.append(self.flow_info[0])
        self.deadline = self.flow_info[4]

    def get_node_state(self):
        node_state = np.zeros([args.node_num, args.node_state_dim])
        curr_node_idx = self.visited_node[-1]
        valid_node = self.neighbors[curr_node_idx]

        for node in self.graph.nodes.values():
            node_state[node.idx][0] = node.buff_status / node.buff_size
            node_state[node.idx][1] = 1 if node.idx == curr_node_idx else 0
            node_state[node.idx][2] = 1 if node.idx == self.flow_info[1] else 0
            node_state[node.idx][3] = 1 if node.idx in valid_node and self.check_cycle(node.idx) else 0

        return node_state

    def get_edge_state(self):
        edge_state = np.zeros([self.hyper_prd, args.edge_state_dim])
        edge_idx = self.visited_edge[-1]

        for prd in range(self.hyper_prd):
            if prd < self.flow_info[2]:
                pos_of_slot = self.graph.edges[edge_idx].find_slot(prd, self.flow_info[2])
                if pos_of_slot:
                    delay = pos_of_slot[0]
                    valid = 1 if delay < self.deadline else 0
                else:
                    delay = self.hyper_prd * self.slot_num
                    valid = 0
            else:
                delay = self.hyper_prd * self.slot_num
                valid = 0

            edge_state[prd][0] = (prd + 1) / self.hyper_prd
            edge_state[prd][1] = delay / self.deadline
            edge_state[prd][2] = valid

        return edge_state

    def step(self, next_node_idx):
        curr_node_idx = self.visited_node[-1]
        edge_idx = self.node_to_edge.index([curr_node_idx, next_node_idx])

        if self.check_action(self.graph.edges[edge_idx]):
            self.visited_node.append(next_node_idx)
            self.visited_edge.append(edge_idx)
            if next_node_idx == self.flow_info[1]:
                done = 1
                reward = 10
            else:
                done = 0
                reward = 0
        else:
            done = -1
            reward = -10

        node_state = self.get_node_state()

        return done, reward, node_state

    def judge(self, prd):
        edge_idx = self.visited_edge[-1]
        pos_of_slot = self.graph.edges[edge_idx].find_slot(prd, self.flow_info[2])

        if pos_of_slot and self.deadline - pos_of_slot[0] >= 0:
            self.actions[edge_idx] = pos_of_slot
            if len(self.visited_edge) > 1:
                self.deadline -= pos_of_slot[0]
            done = 1
            reward = 1 - pos_of_slot[0] / (self.hyper_prd * self.slot_num)
        else:
            done = -1
            reward = -10

        return done, reward, pos_of_slot

    def check_action(self, edge):
        if edge.end_node.idx in self.visited_node or \
                edge.end_node.buff_status <= 0 or \
                self.check_cycle(edge.end_node.idx):
            return False

        return True

    def check_cycle(self, node_idx):
        visited = self.visited_node.copy()
        dst = self.flow_info[1]

        if node_idx == dst:
            return False
        if self.depth_first_search(node_idx, visited):
            return True

        return False

    def depth_first_search(self, node_idx, visited):
        visited.append(node_idx)
        valid = self.neighbors[node_idx]
        dst = self.flow_info[1]

        for adj_idx in valid:
            if adj_idx == dst:
                return False
            elif adj_idx in visited:
                continue
            elif adj_idx not in visited:
                if not self.depth_first_search(adj_idx, visited):
                    return False

        return True

    def schedule(self, pos_of_slot):
        edge_idx = self.visited_edge[-1]
        self.graph.edges[edge_idx].end_node.buff_status -= self.flow_info[3]
        self.graph.edges[edge_idx].occupy_slot(pos_of_slot)

    def compensate(self):
        for edge_idx in self.visited_edge:
            self.graph.edges[edge_idx].end_node.buff_status += self.flow_info[3]

    def buff_usage(self):
        buff_usage = []
        for node in self.graph.nodes.values():
            buff_usage.append(node.buff_status / node.buff_size)

        return buff_usage

    def slot_usage(self):
        slot_usage = []
        for edge in self.graph.edges.values():
            slot_usage.append(sum(edge.slot_status) / (self.hyper_prd * self.slot_num))

        return slot_usage

    def renew(self):
        self.visited_node.clear()
        self.visited_edge.clear()
        self.actions = {}
        self.deadline = 0

    def reset(self):
        self.renew()
        for node in self.graph.nodes.values():
            node.reset()
        for edge in self.graph.edges.values():
            edge.reset()
