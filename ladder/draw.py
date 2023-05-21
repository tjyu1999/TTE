import numpy as np
from matplotlib import pyplot as plt


def draw():
    y = np.load('plot/loss.npy', allow_pickle=True)
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    # plt.xlim()
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plot/loss.png')
    plt.show()


if __name__ == '__main__':
    draw()