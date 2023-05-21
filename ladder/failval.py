import datetime
import time
import random
import numpy as np
import torch
from ladder import Ladder
from env import Env
from agent import Agent
from param import args


class FailVal:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.node_num = args.node_num
        node_num = args.node_num[random.randint(0, len(args.node_num) - 1)]
        self.flow_num = args.val_flow_num
        self.data = Ladder(node_num, self.flow_num)
        self.data.overall()
        self.env = Env(self.data)
        self.agent = Agent(self.env.adj_mat, self.env.dis_mat, self.env.neighbors, device)

        self.val_times = args.val_times
        self.failure_step = args.failure_step
        self.val_cnt = 0
        self.num = []

    def val_one(self):
        start_time = time.time()
        num = 0
        table = {}

        self.data.overall()
        self.env = Env(self.data)
        self.env.reset()
        self.agent.adj_mat = self.env.adj_mat
        self.agent.dis_mat = self.env.dis_mat
        self.agent.neighbors = self.env.neighbors

        for f_i in range(self.failure_step):
            self.env.load_info(f_i)
            x, e = self.env.r_state()
            x = torch.from_numpy(x).float()
            e = torch.from_numpy(e).float()
            table[f_i] = {}
            s_failed = False

            while True:
                if s_failed:
                    break

                i = self.env.visited_node[-1]
                j = self.agent.r_make_action(i, x, e)
                r_done, reward, r_state = self.env.r_step(j)

                x, e = r_state
                x = torch.from_numpy(x).float()
                e = torch.from_numpy(e).float()

                if r_done == 0:
                    p = self.env.s_state()
                    p = torch.from_numpy(p).float()
                    prd = self.agent.s_make_action(p, self.env.flow_info[2])
                    s_done, reward, pos = self.env.s_step(prd)
                    if s_done == 1:
                        e_i = self.env.n2e.index([i, j])
                        self.env.val_schedule(e_i, pos)
                        table[f_i][e_i] = pos
                    elif s_done == -1:
                        s_failed = True

                elif r_done == 1:
                    p = self.env.s_state()
                    p = torch.from_numpy(p).float()
                    prd = self.agent.s_make_action(p, self.env.flow_info[2])
                    s_done, reward, pos = self.env.s_step(prd)
                    if s_done == 1:
                        e_i = self.env.n2e.index([i, j])
                        self.env.val_schedule(e_i, pos)
                        table[f_i][e_i] = pos
                        num += 1

                    break

                elif r_done == -1:
                    break

            self.env.renew()

        e_i, e_j = self.env.graph.failure()
        redo = self.env.failure_compensate(table, e_i, e_j)
        self.env.adj_mat = self.agent.adj_mat = self.env.graph.adj_mat
        self.env.dis_mat = self.agent.dis_mat = self.env.graph.dis_mat
        self.env.neighbors = self.agent.neighbors = self.env.graph.neighbors
        self.env.n2e = self.env.graph.n2e

        for f_i in range(self.flow_num):
            if f_i < self.failure_step and f_i not in redo:
                continue

            self.env.load_info(f_i)
            x, e = self.env.r_state()
            x = torch.from_numpy(x).float()
            e = torch.from_numpy(e).float()
            s_failed = False

            while True:
                if s_failed:
                    break

                i = self.env.visited_node[-1]
                j = self.agent.r_make_action(i, x, e)
                r_done, reward, r_state = self.env.r_step(j)

                x, e = r_state
                x = torch.from_numpy(x).float()
                e = torch.from_numpy(e).float()

                if r_done == 0:
                    p = self.env.s_state()
                    p = torch.from_numpy(p).float()
                    prd = self.agent.s_make_action(p, self.env.flow_info[2])
                    s_done, reward, pos = self.env.s_step(prd)
                    if s_done == 1:
                        e_i = self.env.n2e.index([i, j])
                        self.env.val_schedule(e_i, pos)
                    elif s_done == -1:
                        s_failed = True

                elif r_done == 1:
                    p = self.env.s_state()
                    p = torch.from_numpy(p).float()
                    prd = self.agent.s_make_action(p, self.env.flow_info[2])
                    s_done, reward, pos = self.env.s_step(prd)
                    if s_done == 1:
                        e_i = self.env.n2e.index([i, j])
                        self.env.val_schedule(e_i, pos)
                        num += 1

                    break

                elif r_done == -1:
                    break

            self.env.renew()

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Val Time: {} |'.format(self.val_cnt + 1),
              'Num: {} |'.format(num),
              'Time: {:.02f}s'.format(time.time() - start_time))
        print('#' * 60)

    def val(self):
        self.agent.eval_model()
        self.agent.load_model()
        for _ in range(self.val_times):
            self.val_one()
        self.save()

    def save(self):
        np.save('../failure/failval.npy', self.num)


def main():
    val = FailVal()
    val.val()


if __name__ == '__main__':
    main()