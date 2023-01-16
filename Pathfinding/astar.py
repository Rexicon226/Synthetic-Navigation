from CompositeEnvironment import Visualizer, Environment
import matplotlib.pyplot as plt
import random
import numpy as np
from Terrain import generator

if __name__ == '__main__':
    seed = random.randint(1, 100000000000)

    x = random.randint(50, 200)
    y = random.randint(50, 200)

    noise_level = 30

    print("({}, {})".format(x, y))

    pic = np.array(generator.generateClean(256, 256, 5, seed, True))
    noisy_pic = np.array(generator.generateNoise(256, 256, 5, noise_level, seed, True))

    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))

    masked = ev.generate()

    vi = Visualizer('../ML/models/synthnav-model-0.pth', pic)

    de_noised_original, loss = vi.dNoise(masked)

    de_noised = vi.thresholdDNoise(de_noised_original, 0.5)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(masked, cmap='plasma_r')
    ax[1].imshow(de_noised, cmap='plasma_r')

    plt.show()
