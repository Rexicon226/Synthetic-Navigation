from Terrain import generator
from Terrain import reverser
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def array2image(x, y, octaves, weight, seed: int = 0, iD: int = 0):
    array = generator.generateClean(x, y, octaves, seed)
    plt.imsave('clean/' + str(iD) + '_image.png', array, cmap='binary')
    array = generator.generateNoise(x, y, octaves, weight,seed)
    plt.imsave('noise/' + str(iD) + '_image.png', array, cmap='binary')


image_count = 100

for i in tqdm(range(image_count)):
    seed = random.randint(1, 10000000000)
    array2image(50, 50, 4, 10, seed, i)

print("done")
