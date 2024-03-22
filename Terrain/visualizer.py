import matplotlib.pyplot as plt

import Terrain.blob as blob
import Terrain.border as border
import Terrain.noiseAdder as noiseAdder
import Terrain.pathcheck as pathcheck
import Terrain.sumPerlin as sumPerlin
from Terrain.timing import FunctionTimer


def visualize(
    x: int,
    y: int,
    octaves: int,
    noiselevel: int,
    progress: bool = False,
    setseed: int = 0,
    show: bool = False,
):
    """
    Simple Function used to quickly visualize multi-view
    """
    pic = pathcheck.path(x, y, octaves, progress, setseed)
    noisepic = noiseAdder.addNoise(pic, noiselevel)
    borderpic = border.bordercheck(pic)
    f = FunctionTimer("Island Check")
    perlinnoise = sumPerlin.correctNoiseMaps(x, y, octaves, noiselevel / 100, setseed)
    islands = blob.blobs(pic)
    noiseislands = blob.blobs(noisepic)
    totalnoise = noiseAdder.addNoise(perlinnoise, noiselevel)
    print("      Pic Islands: " + str(islands))
    print("      NoisePic Islands: " + str(noiseislands))

    f.stop()
    if noisepic == perlinnoise:
        print("same")

    if show:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
        axes[0][0].imshow(pic, cmap="binary")
        axes[0][1].imshow(totalnoise, cmap="binary")
        axes[1][0].imshow(pic, cmap="winter_r")
        axes[1][0].imshow(borderpic, cmap="binary", alpha=0.8)
        axes[1][1].imshow(borderpic, cmap="binary")
        axes[0][2].imshow(perlinnoise, cmap="binary")
        axes[1][2].imshow(noisepic, cmap="binary")
        plt.show()

    return pic, totalnoise, perlinnoise, noisepic


if __name__ == "__main__":
    visualize(100, 100, 20, 10, True, 12309)
