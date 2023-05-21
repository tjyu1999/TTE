import os
import datetime
import time
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ladder import Ladder
from env import Env
from agent import Agent
from param import args


writer = SummaryWriter()
torch.manual_seed(args.seed)


class Trainer:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.node_num = args.node_num
        node_num = args.node_num[random.randint(0, len(args.node_num) - 1)]
        self.flow_num = args.train_flow_num
        self.data = Ladder(node_num, self.flow_num)
        self.data.overall()
        self.env = Env(self.data)
        self.agent = Agent(self.env.adj_mat, self.env.dis_mat, self.env.neighbors, device)

        self.episodes = args.episodes
        self.update_step = args.update_step
        self.train_cnt = 0
        self.loss = []

    def train_one(self):
        start_time = time.time()
        num = 0

        node_num = self.node_num[random.randint(0, len(self.node_num) - 1)]
        self.data = Ladder(node_num, self.flow_num)
        self.data.overall()
        self.env = Env(self.data)
        self.env.reset()
        self.agent.adj_mat = self.env.adj_mat
        self.agent.dis_mat = self.env.dis_mat
        self.agent.neighbors = self.env.neighbors

        for flow_i in range(self.flow_num):
            s_failed = False
            actions = []
            self.env.load_info(flow_i)
            x, e = self.env.r_state()
            x = torch.from_numpy(x).float()
            e = torch.from_numpy(e).float()

            while True:
                s = [x, e]
                i = self.env.visited_node[-1]
                j = self.agent.r_make_action(i, x, e)
                r_done, reward, r_state = self.env.r_step(j)

                x, e = r_state
                x = torch.from_numpy(x).float()
                e = torch.from_numpy(e).float()
                s_ = [x, e]
                reward = torch.FloatTensor([reward])[0]
                d = 0 if r_done == 0 else 1
                self.agent.r_push(s, [i, j], reward, s_, d)

                if r_done == 0:
                    if not s_failed:
                        p = self.env.s_state()
                        p = torch.from_numpy(p).float()
                        prd = self.agent.s_make_action(p, self.env.flow_info[2])
                        s_done, _reward, pos = self.env.s_step(prd)

                        _reward = torch.FloatTensor([_reward])[0]
                        self.agent.s_push(p, prd, _reward)
                        if s_done == 1:
                            e_i = self.env.n2e.index([i, j])
                            actions.append([e_i, pos])
                        elif s_done == -1:
                            s_failed = True
                    continue

                elif r_done == 1:
                    if not s_failed:
                        p = self.env.s_state()
                        p = torch.from_numpy(p).float()
                        prd = self.agent.s_make_action(p, self.env.flow_info[2])
                        s_done, _reward, pos = self.env.s_step(prd)

                        _reward = torch.FloatTensor([_reward])[0]
                        self.agent.s_push(p, prd, _reward)
                        if s_done == 1:
                            e_i = self.env.n2e.index([i, j])
                            actions.append([e_i, pos])
                            self.env.train_schedule(actions)
                            num += 1
                    break

                elif r_done == -1:
                    break

            self.env.renew()

            if (flow_i + 1) % self.update_step == 0:
                r_loss, s_loss = self.agent.learn()
                loss = r_loss + s_loss
                self.loss.append(loss.item())
                writer.add_scalar('r_loss', r_loss, self.train_cnt)
                writer.add_scalar('s_loss', s_loss, self.train_cnt)
                writer.add_scalar('loss', loss, self.train_cnt)
                self.train_cnt += 1

                print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                      'Epoch: {} |'.format(self.train_cnt),
                      'Loss: {:.04f}'.format(loss))
                print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                      '-' * 30)

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Num: {} |'.format(num),
              'Time: {:.02f}s'.format(time.time() - start_time))
        print('#' * 50)

    def train(self):
        self.agent.train_model()
        for _ in range(self.episodes):
            self.train_one()
            self.save()

    def save(self):
        np.save('plot/loss.npy', self.loss)


writer.close()


def main():
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('plot'):
        os.mkdir('plot')
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()