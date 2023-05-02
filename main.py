import os
import datetime
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ladder import Ladder
from env import Env
from agent import Agent
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
        self.agent = Agent(adj_mat, dist_mat, neighbors, self.device)
        self.scheduler = Scheduler(self.device)

        self.train_cnt = 0
        self.val_cnt = 0
        self.record = {'router loss': [], 'scheduler loss': [], 'train num': [], 'val num': []}

    def train_one_episode(self):
        start_time = time.time()
        num = 0
        self.data.gen_all_data()
        self.env = Env(self.data)
        self.env.reset()

        adj_mat = torch.from_numpy(self.env.adj_mat).float().to(self.device)
        dist_mat = torch.from_numpy(self.env.dist_mat).float().to(self.device)

        for flow_idx in range(len(self.data.flow_info)):
            self.env.get_info(flow_idx)
            node_state = self.env.get_node_state()
            node_state = torch.from_numpy(node_state).float().to(self.device)

            while True:
                state = node_state
                curr_node_idx = self.env.visited_node[-1]
                next_node_idx = self.agent.select_action(curr_node_idx, node_state, adj_mat, dist_mat)
                done, reward, node_state = self.env.step(next_node_idx)

                reward = torch.FloatTensor([reward])[0]
                node_state = torch.from_numpy(node_state).float().to(self.device)
                d = 0 if done == 0 else 1
                self.agent.store_pair(state, [curr_node_idx, next_node_idx], reward, node_state, d)

                if done == 0:
                    flow_prd = self.env.flow_info[2]
                    edge_state = self.env.get_edge_state()
                    edge_state = torch.from_numpy(edge_state).float().to(self.device)

                    prd, log_prob = self.scheduler.slot_assignment(edge_state, flow_prd)
                    done_, reward_, pos_of_slot = self.env.judge(prd)
                    self.scheduler.store(log_prob, reward_)

                    if done_ == 1:
                        self.env.schedule(pos_of_slot)
                        continue
                    elif done_ == -1:
                        self.env.compensate()
                        break

                elif done == 1:
                    flow_prd = self.env.flow_info[2]
                    edge_state = self.env.get_edge_state()
                    edge_state = torch.from_numpy(edge_state).float().to(self.device)

                    prd, log_prob = self.scheduler.slot_assignment(edge_state, flow_prd)
                    done_, reward_, pos_of_slot = self.env.judge(prd)
                    self.scheduler.store(log_prob, reward_)

                    if done_ == 1:
                        self.env.schedule(pos_of_slot)
                        num += 1
                        break
                    elif done_ == -1:
                        self.env.compensate()
                        break

                elif done == -1:
                    break

            self.env.renew()

        router_loss = self.agent.learn()
        scheduler_loss = self.scheduler.learn()

        self.record['router loss'].append(router_loss.item())
        self.record['scheduler loss'].append(scheduler_loss.item())
        self.record['train num'].append(num)
        writer.add_scalar('router loss', router_loss, self.train_cnt)
        writer.add_scalar('scheduler loss', scheduler_loss, self.train_cnt)
        writer.add_scalar('number', num, self.train_cnt)
        self.train_cnt += 1

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Episode: {} |'.format(self.train_cnt),
              'Time: {:.02f}s'.format(time.time() - start_time))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Router Loss: {:.04f} |'.format(router_loss.item()),
              'Scheduler Loss: {:.04f}'.format(scheduler_loss.item()))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Num: {}'.format(num))
        print('#' * 60)

    def train(self):
        for _ in range(args.episodes):
            self.train_one_episode()

    @torch.no_grad()
    def val_one_episode(self):
        pass

    def val(self):
        self.data = Ladder(flow_num=args.val_flow_num)
        for _ in range(args.val_times):
            self.val_one_episode()

        avg_num = sum(self.record['val num']) / len(self.record['val num'])
        print(avg_num)

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