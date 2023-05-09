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

        self.data = Ladder(args.train_flow_num)
        self.data.gen_all_data()
        self.env = Env(self.data)

        adj_mat = self.env.graph.adj_mat
        dist_mat = self.env.graph.dist_mat
        neighbors = self.env.graph.neighbors
        self.router = Router(adj_mat, dist_mat, neighbors, self.device)
        self.scheduler = Scheduler(self.device)

        self.train_cnt = 0
        self.val_cnt = 0
        self.record = {'router loss': [], 'scheduler loss': []}
        self.val_record = []

    def train_one_episode(self):
        start_time = time.time()
        self.router.model.train()
        self.scheduler.model.train()

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

            schedule_failed = False

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
                    if not schedule_failed:
                        flow_prd = self.env.flow_info[2]
                        edge_state = self.env.get_edge_state()
                        edge_state = torch.from_numpy(edge_state).float().to(self.device)

                        prd = self.scheduler.slot_assignment(edge_state, flow_prd)
                        done_, reward_, pos_of_slot = self.env.judge(prd)
                        reward_ = torch.FloatTensor([reward_])[0].to(self.device)
                        self.scheduler.store_pair(edge_state, prd, reward_)

                        if done_ == 1:
                            self.env.schedule(pos_of_slot)
                        elif done_ == -1:
                            schedule_failed = True
                            self.env.slot_compensate()
                    continue

                elif done == 1:
                    if not schedule_failed:
                        router_num += 1
                        flow_prd = self.env.flow_info[2]
                        edge_state = self.env.get_edge_state()
                        edge_state = torch.from_numpy(edge_state).float().to(self.device)

                        prd = self.scheduler.slot_assignment(edge_state, flow_prd)
                        done_, reward_, pos_of_slot = self.env.judge(prd)
                        reward_ = torch.FloatTensor([reward_])[0].to(self.device)
                        self.scheduler.store_pair(edge_state, prd, reward_)

                        if done_ == 1:
                            scheduler_num += 1
                            self.env.schedule(pos_of_slot)
                        elif done_ == -1:
                            self.env.slot_compensate()
                    break

                elif done == -1:
                    self.env.buff_compensate()
                    self.env.slot_compensate()
                    break

            self.env.renew()

        router_loss = self.router.learn()
        scheduler_loss = self.scheduler.learn()

        self.record['router loss'].append(router_loss.item())
        self.record['scheduler loss'].append(scheduler_loss.item())
        writer.add_scalar('router loss', router_loss, self.train_cnt)
        writer.add_scalar('scheduler loss', scheduler_loss, self.train_cnt)

        self.router.reset_memory()
        self.scheduler.reset_memory()
        self.train_cnt += 1

        is_best = True if router_loss.item() == min(self.record['router loss']) else False
        self.router.save_model(is_best)
        is_best = True if scheduler_loss.item() == min(self.record['scheduler loss']) else False
        self.scheduler.save_model(is_best)

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Epoch: {} |'.format(self.train_cnt),
              'Time: {:.02f}s |'.format(time.time() - start_time),
              'Num: {}/{} |'.format(router_num, scheduler_num),
              'Loss: {:.04f}/{:.04f}'.format(router_loss, scheduler_loss))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              '-' * 70)

        buff_usage = self.env.buff_usage()
        slot_usage = self.env.slot_usage()
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Buff Usage: ', end='')
        for i in buff_usage:
            print('{:.02f}'.format(i), end='|')
        print()
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Slot Usage: ', end='')
        for i in slot_usage:
            print('{:.02f}'.format(i), end='|')
        print()
        print('#' * 190)

    def train(self):
        for _ in range(args.episodes):
            self.train_one_episode()

    @torch.no_grad()
    def val_one_episode(self):
        start_time = time.time()
        self.router.model.eval()
        self.scheduler.model.eval()
        self.router.load_model(is_best=True)
        self.scheduler.load_model(is_best=True)

        val_num = 0

        self.data = Ladder(flow_num=args.val_flow_num)
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
                curr_node_idx = self.env.visited_node[-1]
                next_node_idx = self.router.select_action(curr_node_idx, node_state, adj_mat, dist_mat)
                done, reward, node_state = self.env.step(next_node_idx)
                node_state = torch.from_numpy(node_state).float().to(self.device)

                if done == 0:
                    flow_prd = self.env.flow_info[2]
                    edge_state = self.env.get_edge_state()
                    edge_state = torch.from_numpy(edge_state).float().to(self.device)

                    prd = self.scheduler.slot_assignment(edge_state, flow_prd)
                    done_, reward_, pos_of_slot = self.env.judge(prd)

                    if done_ == 1:
                        self.env.schedule(pos_of_slot)
                    elif done_ == -1:
                        scheduler_failed = True
                        self.env.slot_compensate()

                elif done == 1:
                    flow_prd = self.env.flow_info[2]
                    edge_state = self.env.get_edge_state()
                    edge_state = torch.from_numpy(edge_state).float().to(self.device)

                    prd = self.scheduler.slot_assignment(edge_state, flow_prd)
                    done_, reward_, pos_of_slot = self.env.judge(prd)

                    if done_ == 1:
                        val_num += 1
                        self.env.schedule(pos_of_slot)
                    elif done_ == -1:
                        self.env.slot_compensate()
                    break

                elif done == -1:
                    self.env.buff_compensate()
                    self.env.slot_compensate()
                    break

                if not scheduler_failed:
                    continue
                elif scheduler_failed:
                    break

            self.env.renew()
            if done == -1 or scheduler_failed:
                break

        self.val_record.append(val_num)
        self.val_cnt += 1

        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Val Time: {} |'.format(self.val_cnt),
              'Num: {} |'.format(val_num),
              'Time: {:.02f}s'.format(time.time() - start_time))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              '-' * 50)

        buff_usage = self.env.buff_usage()
        slot_usage = self.env.slot_usage()
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Buff Usage: ', end='')
        for i in buff_usage:
            print('{:.02f}'.format(i), end='|')
        print()
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Slot Usage: ', end='')
        for i in slot_usage:
            print('{:.02f}'.format(i), end='|')
        print()
        print('#' * 190)

    def val(self):
        for _ in range(args.val_times):
            self.val_one_episode()
        print(sum(self.record['val num']) / len(self.record['val num']))

    def save_record(self):
        if not os.path.exists('record'):
            os.makedirs('record')
        for key in self.record.keys():
            np.save(f'record/{key}.npy', self.record[key])
        np.save('record/val_record.npy', self.val_record)


writer.close()


def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    trainer = Trainer()
    trainer.train()
    trainer.val()
    trainer.save_record()


if __name__ == '__main__':
    main()