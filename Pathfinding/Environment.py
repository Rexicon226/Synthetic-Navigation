"""Pathfinding environment that different algorithms interact with"""
import copy

import numpy as np
import enum
from typing import Tuple
from math import sqrt
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys


# adding Folder_2 to the system path
sys.path.insert(0, "..")

OPEN = 0
WALL = 1
VISITED = 0.5
CURRENT = 2

epsilon = 0.1

BLOCKED_PENALTY = -0.25
MOVEMENT_PENALTY = -0.04
OUTOFBOUNDS_PENALTY = -0.25
WIN_REWARD = 1


class ACTS(enum.Enum):
    U = (0, -1)  # up
    D = (0, 1)  # down
    L = (-1, 0)  # left
    R = (1, 0)  # right


def circleUpdate(base, add, radius, coordinates: Tuple[int, int]):
    for y, ny in enumerate(base):
        for x, nx in enumerate(base):
            distance = sqrt((coordinates[0] - x) * (coordinates[0] - x) + (coordinates[1] - y) * (coordinates[1] - y))
            if distance < radius:
                base[y][x] = add[y][x]

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
        y, x = pmap.shape
        self.radius = radius
        self.start = (0, 0)
        self.goal = (x - 1, y - 1)
        self.takenPath = []
        self.pos = self.start

        self.pmap = pmap
        self.amap = amap
        self.intmap = copy.deepcopy(self.pmap)  # interpolated map (combination)
        # map for viewing

        self.reward = 0
        self.is_won = False

    def viewing_state(self):
        axes[0].imshow(self.pmap)
        axes[1].imshow(self.amap)
        vmap = copy.deepcopy(self.intmap)
        vmap[self.pos[0]][self.pos[1]] = 3
        for x, y in self.takenPath:
            vmap[x][y] = 2

        axes[2].imshow(vmap)

    def move(self, action: ACTS):
        self.penalty = 0
        self.takenPath.append(self.pos)
        newpos = tuple(map(lambda i, j: i + j, self.pos, action.value))
        if newpos[0] > 256 or newpos[0] < 0 or newpos[1] > 256 or newpos[1] < 0:
            self.penalty += OUTOFBOUNDS_PENALTY
        elif self.amap[newpos[0]][newpos[1]] == 1:  # actor has run into a wall
            self.penalty += BLOCKED_PENALTY
        else:
            self.penalty += MOVEMENT_PENALTY
            self.pos = newpos
            circleUpdate(self.intmap, self.amap, self.radius, self.pos)

        if self.pos == self.goal:
            self.penalty += WIN_REWARD
            self.is_won = True


def update(i):
    act = random.choice([ACTS.U, ACTS.D, ACTS.L, ACTS.R])
    env.move(act)
    env.viewing_state()


if __name__ == "__main__":
    import Terrain.pathcheck as pathcheck

    amap = pathcheck.path(256, 256, 3, setseed=1239082)
    pmap = pathcheck.path(256, 256, 3, False, 1)
    # amap, pmap = sumPerlin.thresholdedNoiseMaps(256,256,3,0.40)
    env = Environment(np.array(pmap), np.array(amap), 16)
    fig, axes = plt.subplots(1, 3)
    anim = FuncAnimation(fig, update, frames=20, interval=50)
    plt.show()
    print("done")
