import copy
import sys
from random import choices

import matplotlib.pyplot as plt

import pathcheck


def addnoise(pic: list[list[(int)]], weight: int):
    """
    This function takes in an array and adds random noise to it depending on the weights set.

    Parameters
    ----------
    pic : list[list[1,0]]
        2 dimensional binary array of the "terrain". Typically generated with perlin noise.

    weight : int
        Ranges from 0 to 100!
        A value that determines the weight of the noise. More mean a greater amount of noise.
        Given weights [0.5, 0.5] for an array [0, 1], the output has a 50% chance of being 1 or 0

        In this example, x will be our weight.
        Given weights[x / 100, 1 - (x / 100)] for an array [0, 1], the output hsa a X% chance of being 1 or 0

    Returns
    -------
    noisepic : list[list[0,1]]
        2 dimensional binary array of terrain but with noise introduced. This is too mimic the
        corruption and "spotty" nature of real world data
    """
    if 0 <= weight <= 100:
        pass
    else:
        sys.exit("Weight is out of range!\nMake sure it is withing the range (0, 100)")
    noisepic = copy.deepcopy(pic)
    for i in range(len(noisepic)):
        for j in range(len(noisepic[i])):
            choice = choices([0, 1], [weight / 100, 1 - (weight / 100)])
            if choice[0] == 1:
                pass
            else:
                if noisepic[i][j] == 1:
                    noisepic[i][j] = 0
                else:
                    noisepic[i][j] = 1
    return noisepic


if __name__ == '__main__':
    pic = pathcheck.path(50, 50, 5, True, 1347242)
    noisepic = addnoise(pic, 5)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].imshow(pic, cmap='Greys')
    axes[1].imshow(noisepic, cmap='Greys')
    plt.show()
