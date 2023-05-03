import os
import numpy as np
from matplotlib import pyplot as plt


def draw():
    types = ['router loss', 'scheduler loss', 'router num', 'scheduler num']

    for data in types:
        y = np.load(f'record/{data}.npy', allow_pickle=True)
        x = np.arange(len(y))
        plt.plot(x, y, 'r')
        plt.grid(True)
        plt.savefig(f'plot/{data}.jpg')


if __name__ == '__main__':
    if not os.path.exists('plot'):
        os.makedirs('plot')
    draw()