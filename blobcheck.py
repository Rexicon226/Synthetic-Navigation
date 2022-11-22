import copy
from collections import deque

import matplotlib.pyplot as plt

import border
import noiseadder
import pathcheck
from timers import FunctionTimer


def blobs(pic: list[list[int]]):
    """
    Calculates the number of "islands" in the given array

    Rules:
    The edges of the graph count as "walls" so that even is an island is cut off
    it is still counted.

    Parameters
    ----------
    pic : list[list[1,0]]
        2 dimensional binary array of the "terrain". Typically generated with perlin noise.

    Returns
    -------
    islands, oceans : int,int
        number of regions with a value of 0, number of regions with a value of 1"""

    row = [-1, -1, -1, 0, 1, 0, 1, 1]
    col = [-1, 1, 0, -1, -1, 1, 0, 1]

    def isSafe(mat, x, y, processed):
        return (x >= 0 and x < len(processed)) and (y >= 0 and y < len(processed[0])) and \
               mat[x][y] == 1 and not processed[x][y]

    def BFS(mat, processed, i, j):
        # create an empty queue and enqueue source node
        q = deque()
        q.append((i, j))

        # mark source node as processed
        processed[i][j] = True

        # loop till queue is empty
        while q:
            # dequeue front node and process it
            x, y = q.popleft()

            # check for all eight possible movements from the current cell
            # and enqueue each valid movement
            for k in range(len(row)):
                # skip if the location is invalid, or already processed, or has water
                if isSafe(mat, x + row[k], y + col[k], processed):
                    # skip if the location is invalid, or it is already
                    # processed, or consists of water
                    processed[x + row[k]][y + col[k]] = True
                    q.append((x + row[k], y + col[k]))

    def countIslands(mat):
        # base case
        if not mat or not len(mat):
            return 0

        # `M × N` matrix
        (M, N) = (len(mat), len(mat[0]))

        # stores if a cell is processed or not
        processed = [[False for x in range(N)] for y in range(M)]

        island = 0
        for i in range(M):
            for j in range(N):
                # start BFS from each unprocessed node and increment island count
                if mat[i][j] == 1 and not processed[i][j]:
                    BFS(mat, processed, i, j)
                    island = island + 1

        return island

    islandpic = copy.deepcopy(pic)
    oceanpic = copy.deepcopy(pic)
    for y in range(len(oceanpic) - 1):
        oceanpic[y] = [0 if (k == 1) else 1 for k in oceanpic[y]]

    islands = countIslands(islandpic)
    oceans = countIslands(oceanpic)

    return islands


def visualize(x: int, y: int, octaves: int, noiselevel: int, progress: bool = False, setseed: int = 0):
    """
    Simple Function used to quickly visualize multi-view
    """
    pic = pathcheck.path(x, y, octaves, progress, setseed)
    noisepic = noiseadder.addnoise(pic, noiselevel)
    borderpic = border.bordercheck(pic)
    f = FunctionTimer("Island Check")
    islands = blobs(pic)
    noiseislands = blobs(noisepic)
    print("Pic Islands: " + str(islands))
    print("NoisePic Islands: " + str(noiseislands))

    f.stop()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    axes[0][0].imshow(pic, cmap='binary')
    axes[0][1].imshow(noisepic, cmap='binary')
    axes[1][0].imshow(pic, cmap='winter_r')
    axes[1][0].imshow(borderpic, cmap='binary', alpha=0.8)
    axes[1][1].imshow(borderpic, cmap='binary')
    plt.show()


if __name__ == "__main__":
    visualize(300, 300, 20, 20, True, 120938561)
