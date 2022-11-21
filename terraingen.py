import numpy as np
from perlin_noise import PerlinNoise


def terrain(size, randseed):
    noise = PerlinNoise(octaves=size, seed=randseed)
    xpix, ypix = 15, 15
    pic = [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
    totalblack = 0
    for i in range(xpix):
        for j in range(ypix):
            pic[i][j] = np.floor(pic[i][j])
            if pic[i][j] == -1:
                totalblack += 1
    return pic

