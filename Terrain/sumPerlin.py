from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from perlin_noise import PerlinNoise


def noiseMaps(x: int, y: int, octaves1, octaves2, difference: float, seed: Optional[int] = 0,
              seedDifference: Optional[int] = 1):
    """function that generates two maps with a "difference" in between
    works by generating two perlin noise maps and then adding them and returning both the sum and one individual
    Parameters
    ----------
    x : int
        x size of the maps
    y : int
        y size of the maps
    octaves1 : int
        number of octaves in the base layer of both maps
    octaves2 : int
        number of octaves in the second layer of the second map
    difference : float
        percentage difference between the maps (opacity of second layer of second map)
    seed : Optional[int]
        random seed for number generation
    seedDifference : Optional[int]
        difference between the two seeds
    """

    map1 = PerlinNoise(octaves1, seed)
    map2 = PerlinNoise(octaves2, seed + seedDifference)
    discrete1 = [[map1([i / x, j / y]) for j in range(y)] for i in range(x)]
    discrete2 = [[(difference * map2([i / x, j / y])) + (1 - difference) * (discrete1[i][j]) for j in range(y)] for i in
                 range(x)]

    return discrete1, discrete2


def thresholdedNoiseMaps(x: int, y: int, octaves1, octaves2, difference: float, seed: Optional[int] = 0,
                         seedDifference: Optional[int] = 1):
    """same as noiseMaps only it gets thresholded"""
    d1, d2 = noiseMaps(x, y, octaves1, octaves2, difference, seed, seedDifference)
    td1 = [[int(np.floor(x)) for x in d1[y]] for y in range(y)]
    td2 = [[int(np.floor(x)) for x in d2[y]] for y in range(y)]

    return td1, td2


if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    d1, d2 = thresholdedNoiseMaps(100, 100, 8, 8, 0.10)
    axes[0].imshow(d1, cmap="Greys")
    axes[1].imshow(d2, cmap="Greys")
    plt.show()
