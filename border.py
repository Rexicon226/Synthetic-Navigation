import copy
import time

import numpy as np

import terraingen


def bordercheck(pic: list[list[int]]):
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
    print('--- Starting Border Find ---')
    start_time = time.time()

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

    print("Pic: " + str(pic))
    print("Border: " + str(borderpic))
    print("%s seconds of processing" % np.round(time.time() - start_time, 2))
    print("---Done Border Check---")
    return borderpic


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pic = bordercheck(terraingen.terrain(20, 20, 4))
    plt.imshow(pic, cmap="Greys")
    plt.show()
