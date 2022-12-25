import Terrain.terraingen as terraingen
import Terrain.sumPerlin as sumPerlin
import Terrain.noiseadder as noiseadder


def generateClean(x, y, octaves, seed, progress: bool = False):
    array = terraingen.terrain(x, y, octaves, progress, seed)
    return array


def generateNoise(x, y, octaves, weight, seed, progress: bool = False):
    array = terraingen.terrain(x, y, octaves, progress, seed)
    array = sumPerlin.thresholdedNoiseMaps(x, y, octaves, weight / 100, seed)
    array = noiseadder.addnoise(array, weight)
    return array
