import random
import sys
import time

import numpy as np

import terraingen


def path(size, octaves, setseed=0):
    """TODO: @Rexicon266 add comments and docstring"""
    col = size
    row = size

    def isPath(arr):
        arr[0][0] = 1

        for i in range(1, row):
            if arr[i][0] != -1:
                arr[i][0] = arr[i - 1][0]

        for j in range(1, col):
            if arr[0][j] != -1:
                arr[0][j] = arr[0][j - 1]

        for i in range(1, row):
            for j in range(1, col):
                if arr[i][j] != -1:
                    arr[i][j] = max(arr[i][j - 1],
                                    arr[i - 1][j])

        return arr[row - 1][col - 1] == 1

    solved = False
    failedSeeds = []
    start_time = time.time()
    while not solved:
        if setseed != 0 and len(failedSeeds) == 0:
            seed = setseed
        else:
            seed = random.randint(1, 10000000)
        if failedSeeds.count(seed) == 0:
            solved = isPath(terraingen.terrain(octaves, seed, size, size))
            sys.stdout.write("\rChecking seed: " + str(seed) + ", Number: " + str(len(failedSeeds) + 1))
            sys.stdout.flush()
            if solved:
                print(f'\nWorking Seed: ' + str(seed))
                print(f'Failed Seeds: ' + str(len(failedSeeds)))
                print("%s seconds of processing" % np.round(time.time() - start_time, 2))
                print("--- Done Path Checking ---")
            failedSeeds.append(seed)
    A = terraingen.terrain(octaves, seed, size, size)
    for i in range(len(A)):
        A[i] = list(map(int, A[i]))
        A[i] = [abs(ele) for ele in A[i]]
    return A


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = path(10, 2)
    plt.imshow(pic, cmap="Greys")
    plt.show()
