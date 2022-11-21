import numpy as np
from perlin_noise import PerlinNoise


def terrain(octaves: int, seed: int, x: int, y: int):
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
        thresholded list of smooth noise
    """
    noise = PerlinNoise(octaves=octaves, seed=seed)
    xpix, ypix = x, y
    pic = [[np.floor(noise([i / xpix, j / ypix])) for j in range(xpix)] for i in range(ypix)]

    return pic
