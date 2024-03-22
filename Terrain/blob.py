import generator
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

import sumPerlin
import noiseAdder

def blobs(array: Union[list[list[int]], np.ndarray]) -> int:
    """
    Calculates the number of "islands" in the given array

    Rules:
    The edges of the graph count as "walls" so that even is an island is cut off
    it is still counted.

    Parameters
    ----------
    array : list[list[1,0]] / np.ndarray
        2 dimensional binary array of the "terrain" or a `NDArray` of the same shape.

    Returns
    -------
    islands: int
        number of regions with a value of 0
    """
    # Create a copy of the array to mark visited cells
    visited = [[False for _ in row] for row in array]

    # Initialize the island count to 0
    island_count = 0

    for i in range(len(array)):
        for j in range(len(array[i])):
            print("here")
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


if __name__ == "__main__":
    size = 256
    octaves = 5
    noise_level = 30
    seed = 12308
    clean_map = generator.generateClean(size, size, octaves, noise_level, seed, True)
    noise_map = sumPerlin.correctNoiseMaps(clean_map, size, size, octaves, noise_level / 100, seed)
    noise_map = noiseAdder.addNoise(noise_map, noise_level / 2)

    clean_map_islands = blobs(clean_map)
    # noise_map_islands = blobs(noise_map)

    print(clean_map_islands)

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Clean")
    axes[0].imshow(clean_map)

    axes[1].set_title("Noisy")
    axes[1].imshow(noise_map)

    # print("Clean map islands: {}, Noisy map islands: {}".format(clean_map_islands, noise_map_islands))

    plt.show()
