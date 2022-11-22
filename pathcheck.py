import os
import random
import sys

import terraingen
from timers import FunctionTimer


def path(x: int, y: int, octaves: int, progress: bool = False, setseed: int = 0):
    """
    Parameters
    ----------
    Checks if the generated terrain map has a path
    from the top left square to the bottom right square

    If no path, a new terrain map will be generated until one is found

    (I know it's not efficient but random perlin noise moment)

    x : int
        x size of the graph

    y : int
        y size of the graph

    octaves : int
        Changes the complexity of the perlin noise.
        (runtime will increase drastically, don't increase over 7-10)
    Progress : bool
        True: will display progress bar (not recommended for uses were time is unknown)
        False: Default, will not display progress bar
    setseed : int
        Forces a seed into the perlin noise generation, useful for debugging.
        Optional value not needed.
        DONT PASS 0 AS THE SET SEED WILL NOT RUN
        (setseed needs to be specified in teh function call)

    Returns
    -------
    pic : list[list[0,1]]
        list that has a path from TL (top left) to BR (bottom right)
    """
    col = x
    row = y

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

    def clear():
        os.system('cls')

    solved = False
    failedSeeds = []
    f = FunctionTimer("Path Processing")
    while not solved:
        if setseed != 0:
            seed = setseed
            print("Set Seed: " + str(seed))
            solved = True
        else:
            seed = random.randint(1, x * y * 1000)
            clear()
        if failedSeeds.count(seed) == 0 and setseed == 0:
            solved = isPath(terraingen.terrain(x, y, octaves, True, seed=seed))
            sys.stdout.write("\rChecking seed: " + str(seed) + ", Number: " + str(len(failedSeeds) + 1))
            sys.stdout.flush()
            if solved:
                print(f'\nWorking Seed: ' + str(seed))
                print(f'Failed Seeds: ' + str(len(failedSeeds)))
            failedSeeds.append(seed)
    A = terraingen.terrain(x, y, octaves, progress, seed=seed)
    for i in range(len(A)):
        A[i] = list(map(int, A[i]))
        A[i] = [abs(ele) for ele in A[i]]
    f.stop()
    return A



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = path(100, 100, 10, True)
    plt.imshow(pic, cmap="Greys")
    plt.show()
