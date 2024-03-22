import copy

import Terrain.terraingen as terraingen
from Terrain.timing import FunctionTimer
from typing import Union
import numpy as np


def bordercheck(pic: Union[list[list[int]], np.ndarray]):
    """Returns a modified array where the borders of the blobs are ``0`` and everything else is ``1``.

    Note: The borders are "Inclusive" the edge pixels of the blobs are counted, nothing that isn't blob
    is generated. Overlaying the border and pic graphs would result in the same size.

    Parameters
    ----------
    pic : list[list[{1,0}]]
        2 dimensional binary array of the "terrain". Typically generated with perlin noise.

    Returns
    -------
    borderpic : list[list[{1,0}]]
    """
    f = FunctionTimer("Border Check")

    for i in range(len(pic)):
        pic[i] = [abs(ele) for ele in pic[i]]
    borderpic = copy.deepcopy(pic)  # create a deep copy so that stuff doesn't get messed up

    for x in range(len(pic)):
        for y in range(len(pic)):
            if pic[x][y] == 1:
                mx = 0
                mn = 0
                if y < len(pic) - 1:
                    mx += 1
                    mn += pic[x][y + 1]
                if y > 0:
                    mx += 1
                    mn += pic[x][y - 1]
                if x < len(pic) - 1:
                    mx += 1
                    mn += pic[x + 1][y]
                if x > 0:
                    mx += 1
                    mn += pic[x - 1][y]

                if mx != mn:
                    borderpic[x][y] = 1
                else:
                    borderpic[x][y] = 0

            else:
                borderpic[x][y] = 0
                # borderpic[x][y] = mx-1

    f.stop()
    return borderpic


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = terraingen.terrain(100, 100, 4, True)
    terrainpic = bordercheck(pic)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].imshow(pic, cmap="Greys")
    axes[1].imshow(terrainpic, cmap="binary")
    plt.show()
