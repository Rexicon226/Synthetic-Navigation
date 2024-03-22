import terraingen
import sumPerlin
import Terrain.noiseAdder as noiseAdder


def generateClean(x, y, octaves, seed, progress: bool = False):
    array = terraingen.terrain(x, y, octaves, progress, seed)
    return array


def generateNoise(x, y, octaves, weight, seed, progress: bool = False):
    array = terraingen.terrain(x, y, octaves, progress, seed)
    array = sumPerlin.correctNoiseMaps(x, y, octaves, weight / 100, seed)
    array = noiseAdder.addNoise(array, weight / 2)
    return array
