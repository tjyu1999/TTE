import numpy as np
from matplotlib import pyplot as plt


def draw():
    r_y = np.load('plot/r_loss.npy', allow_pickle=True)
    x = np.arange(len(r_y))
    plt.plot(x, r_y, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plot/r_loss.png')
    plt.show()

    s_y = np.load('plot/s_loss.npy', allow_pickle=True)
    x = np.arange(len(s_y))
    plt.plot(x, s_y, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plot/s_loss.png')
    plt.show()

    y = np.load('plot/loss.npy', allow_pickle=True)
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plot/loss.png')
    plt.show()


if __name__ == '__main__':
    draw()