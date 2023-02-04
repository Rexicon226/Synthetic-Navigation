import copy
from collections import deque
import Terrain.generator as generator
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def blobs(array: Union[list[list[int]], np.ndarray]) -> int:
    """
    Calculates the number of "islands" in the given array

    Rules:
    The edges of the graph count as "walls" so that even is an island is cut off
    it is still counted.

    Parameters
    ----------
    array : list[list[1,0]]
        2 dimensional binary array of the "terrain". Typically generated with perlin noise.

    Returns
    -------
    islands, oceans : int,int
        number of regions with a value of 0, number of regions with a value of 1"""
    # Create a copy of the array to mark visited cells
    visited = [[False for _ in row] for row in array]

    # Initialize the island count to 0
    island_count = 0

    # Iterate through each cell in the array
    for i in range(len(array)):
        for j in range(len(array[i])):
            # If the cell is land and has not been visited yet, it's part of a new island
            if array[i][j] == 1 and not visited[i][j]:
                island_count += 1
                # Mark all cells in the island as visited
                mark_island(array, visited, i, j)

    return island_count


def mark_island(array, visited, i, j):
    # Mark the current cell as visited
    visited[i][j] = True

    # Check the cells adjacent to the current cell
    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    for x, y in neighbors:
        # If the adjacent cell is land and has not been visited yet, it's part of the same island
        if x >= 0 and x < len(array) and y >= 0 and y < len(array[i]) and array[x][y] == 1 and not visited[x][y]:
            mark_island(array, visited, x, y)


if __name__ == '__main__':
    size = 256
    octaves = 5
    weight = 30
    noisepic = generator.generateNoise(size, size, octaves, weight, True)
    pic = generator.generateClean(size, size, octaves, True)

    pic_islands = blobs(pic)
    noisepic_islands = blobs(noisepic)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pic)
    axes[0].set_title("Clean")
    axes[1].imshow(noisepic)
    axes[1].set_title("Noisy")

    print("Clean pic islands: {}, Noisy pic islands: {}".format(pic_islands, noisepic_islands))

    plt.show()
