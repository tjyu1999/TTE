import os
import numpy as np
from matplotlib import pyplot as plt


def draw():
    y = np.load('record/router_loss.npy', allow_pickle=True)
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    # plt.savefig('plot/router_loss.jpg')
    plt.show()

    y = np.load('record/scheduler_loss.npy', allow_pickle=True)
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    # plt.savefig('plot/scheduler_loss.jpg')
    plt.show()

    y = np.load('record/val_record.npy', allow_pickle=True)
    print(sum(y) / len(y))
    plt.boxplot(y)
    plt.grid(True)
    # plt.savefig('plot/val_record.jpg')
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('plot'):
        os.makedirs('plot')
    draw()