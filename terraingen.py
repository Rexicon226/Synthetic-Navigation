import numpy as np
from perlin_noise import PerlinNoise


def terrain(size, randseed, x, y):
    noise = PerlinNoise(octaves=size, seed=randseed)
    xpix, ypix = x, y
    pic = [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
    for i in range(xpix):
        for j in range(ypix):
            pic[i][j] = np.floor(pic[i][j])
    return pic

