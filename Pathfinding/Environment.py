"""Pathfinding environment that different algorithms interact with"""
import numpy as np
import enum
from typing import List, Tuple
from math import sqrt

OPEN = 0
WALL = 1
VISITED = 0.5
CURRENT = 2

epsilon = 0.1


class ACTS(enum.Enum):
    U = 0  # up
    D = 1  # down
    L = 2  # left
    R = 3  # right


def circleUpdate(base, add, radius, coordinates: Tuple[int, int]):
    for x, nx in enumerate(base):
        for y, ny in enumerate(base):
            distance = sqrt((coordinates[0] - x) * (coordinates[0] - x) + (coordinates[1] - y) * (coordinates[1] - y))
            if distance < radius:
                base[x][y] = add[x][y]

    return base


class Environment:
    def __init__(self, pmap: np.array, amap: np.array, radius):
        """
        Parameters
        ----------
        pmap : np.array
            predicted map

        pmap : np.array
            actual map

        radius : int
            "view distance" of the "robot"
        """
        x, y = pmap.shape
        self.start = (0, 0)
        self.goal = (x - 1, y - 1)

        self.pmap = pmap
        self.amap = amap
        self.intmap = self.pmap  # interpolated map (combination)
        self.vmap = self.intmap  # map for viewing


if __name__ == '__main__':
    zeroes = [[0 for x in range(256)] for y in range(256)]
    ones = [[1 for x in range(256)] for y in range(256)]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(circleUpdate(zeroes, ones, 32, (128, 32)), cmap='binary')
    plt.show()
