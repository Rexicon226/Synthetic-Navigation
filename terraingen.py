import random

import matplotlib.pyplot as plt
import numpy as np
from perlin_noise import PerlinNoise

def terrain(size):
    randseed = random.randint(1, 100000)
    noise = PerlinNoise(octaves=size, seed=randseed)
    xpix, ypix = 100, 100
    pic = [[noise([i / xpix, j / ypix]) for j in range(xpix)] for i in range(ypix)]
    for i in range(xpix):
        for j in range(ypix):
            pic[i][j] = np.floor(pic[i][j])
    plt.imshow(pic, cmap='gray')
    plt.show()

terrain(10)