import os
import datetime
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ladder import Ladder
from env import Env
from router import Router
from scheduler import Scheduler
from param import args


writer = SummaryWriter()
torch.manual_seed(args.seed)


class Trainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = Ladder()
        self.data.gen_all_data()
        self.env = Env(self.data)

        adj_mat = self.env.graph.adj_mat
        dist_mat = self.env.graph.dist_mat
        neighbors = self.env.graph.neighbors
        self.router = Router(adj_mat, dist_mat, neighbors, self.device)
        self.scheduler = Scheduler(self.device)

        self.train_cnt = 0
        self.val_cnt = 0
        self.record = {'router loss': [], 'scheduler loss': [], 'router num': [], 'scheduler num': []}

    def train_one_episode(self):
        start_time = time.time()

        router_num = 0
        scheduler_num = 0

        self.data.gen_all_data()
        self.env = Env(self.data)
        self.env.reset()
        adj_mat = torch.from_numpy(self.env.adj_mat).float().to(self.device)
        dist_mat = torch.from_numpy(self.env.dist_mat).float().to(self.device)

        for flow_idx in range(len(self.data.flow_info)):
            self.env.get_info(flow_idx)
            node_state = self.env.get_node_state()
            node_state = torch.from_numpy(node_state).float().to(self.device)
            scheduler_failed = False

            while True:
                state = node_state
                curr_node_idx = self.env.visited_node[-1]
                next_node_idx = self.router.select_action(curr_node_idx, node_state, adj_mat, dist_mat)
                done, reward, node_state = self.env.step(next_node_idx)

                reward = torch.FloatTensor([reward])[0]
                node_state = torch.from_numpy(node_state).float().to(self.device)
                d = 0 if done == 0 else 1
                self.router.store_pair(state, [curr_node_idx, next_node_idx], reward, node_state, d)

                if done == 0:
                    if not scheduler_failed:
                        flow_prd = self.env.flow_info[2]
                        edge_state = self.env.get_edge_state()
                        edge_state = torch.from_numpy(edge_state).float().to(self.device)

                        prd, value = self.scheduler.slot_assignment(edge_state, flow_prd)
                        done_, reward_, pos_of_slot = self.env.judge(prd)
                        reward_ = torch.FloatTensor([reward_])[0].to(self.device)
                        self.scheduler.store_pair(value, reward_)

                        if done_ == 1:
                            self.env.schedule(pos_of_slot)
                        elif done_ == -1:
                            self.env.compensate()
                            scheduler_failed = True

                    continue

                elif done == 1:
                    router_num += 1
                    if not scheduler_failed:
                        flow_prd = self.env.flow_info[2]
                        edge_state = self.env.get_edge_state()
                        edge_state = torch.from_numpy(edge_state).float().to(self.device)

                        prd, value = self.scheduler.slot_assignment(edge_state, flow_prd)
                        done_, reward_, pos_of_slot = self.env.judge(prd)
                        reward_ = torch.FloatTensor([reward_])[0].to(self.device)
                        self.scheduler.store_pair(value, reward_)

                        if done_ == 1:
                            self.env.schedule(pos_of_slot)
                            scheduler_num += 1
                        elif done_ == -1:
                            self.env.compensate()
                    break

                elif done == -1:
                    break

            self.env.renew()

        router_loss = self.router.learn()
        scheduler_loss = self.scheduler.learn()

        self.record['router loss'].append(router_loss.item())
        self.record['scheduler loss'].append(scheduler_loss.item())
        self.record['router num'].append(router_num)
        self.record['scheduler num'].append(scheduler_num)
        writer.add_scalar('router loss', router_loss, self.train_cnt)
        writer.add_scalar('scheduler loss', scheduler_loss, self.train_cnt)
        writer.add_scalar('router num', router_num, self.train_cnt)
        writer.add_scalar('scheduler num', scheduler_num, self.train_cnt)
        self.train_cnt += 1

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Episode: {} |'.format(self.train_cnt),
              'Num: {}/{} |'.format(router_num, scheduler_num),
              'Time: {:.02f}s'.format(time.time() - start_time))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Router Loss: {:.02f} |'.format(router_loss.item()),
              'Scheduler Loss: {:.02f}'.format(scheduler_loss.item()))
        print('#' * 60)

    def train(self):
        for _ in range(args.episodes):
            self.train_one_episode()

    @torch.no_grad()
    def val_one_episode(self):
        pass

    def val(self):
        pass

    def save(self):
        if not os.path.exists('record'):
            os.makedirs('record')
        for key in self.record.keys():
            np.save(f'record/{key}.npy', self.record[key])


writer.close()


def main():
    trainer = Trainer()
    trainer.train()
    # trainer.val()
    trainer.save()


if __name__ == '__main__':
    main()