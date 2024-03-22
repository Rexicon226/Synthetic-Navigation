import numpy as np
from perlin_noise import PerlinNoise
from tqdm import tqdm


def terrain(x: int, y: int, octaves: int, progress: bool = False, seed: int = 0):
    """generates a random "terrain" map

    Parameters
    ----------
    x : int
        x size of the array
    y : int
        y size of the array
    octaves : int
        octaves for perlin noise
    progress : bool
        True: will display progress bar (not recommended for uses were time is unknown)
        False: Default, will not display progress bar
    seed : int
        There are two optional arguments, you need to specify "seed=" in the function
    Returns
    -------
    pic : list[list[0,1]]
        threshold list of smooth noise
    """

    noise = PerlinNoise(octaves=octaves, seed=seed)
    prange = range(x) if not progress else tqdm(range(x))
    return [[int(np.floor(noise([i / x, j / y]))) for j in range(y)] for i in prange]


def threedterrain(x: int, y: int, octaves: int, progress: bool = False, seed: int = 0):
    """generates a random "terrain" map

    Parameters
    ----------
    x : int
        x size of the array
    y : int
        y size of the array
    octaves : int
        octaves for perlin noise
    progress : bool
        True: will display progress bar (not recommended for uses were time is unknown)
        False: Default, will not display progress bar
    seed : int
        There are two optional arguments, you need to specify "seed=" in the function
    Returns
    -------
    pic : list[list[0,1]]
        threshold list of smooth noise
    """
    multiplier = 1.5
    noise = PerlinNoise(octaves=octaves, seed=seed)
    if progress:
        picarr = [[noise([i / x, j / y]) for j in range(y)] for i in tqdm(range(x))]
        avg = np.average(picarr)
        avg2 = np.abs(avg) * multiplier
        for i in range(len(picarr)):
            for j in range(len(picarr[i])):
                if picarr[i][j] < avg:
                    picarr[i][j] = avg2

        return picarr
    if not progress:
        picarr = [[np.floor(noise([i / x, j / y])) for j in range(y)] for i in range(x)]
        avg = np.average(picarr)
        avg2 = np.abs(avg) * multiplier
        for i in range(len(picarr)):
            for j in range(len(picarr[i])):
                if picarr[i][j] < avg:
                    picarr[i][j] = avg2

        return picarr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = threedterrain(256, 256, 4, True, seed=12308)
    plt.imshow(pic, cmap="Greys")
    plt.show()
