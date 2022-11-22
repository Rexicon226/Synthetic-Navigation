import numpy as np
from perlin_noise import PerlinNoise


def terrain(x: int, y: int, octaves: int, seed: int = 0):
    """generates a random "terrain" map

    Parameters
    ----------
    octaves : int
        octaves for perlin noise
    seed : int
        random seed
    x : int
        x size of the array
    y : int
        y size of the array

    Returns
    -------
    pic : list[list[0,1]]
        threshold list of smooth noise
    """

    noise = PerlinNoise(octaves=octaves, seed=seed)
    pic = [[np.floor(noise([i / x, j / y])) for j in range(y)] for i in range(x)]

    return pic


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = terrain(20, 20, 4)
    plt.imshow(pic, cmap="Greys")
    plt.show()
