"""Generator used to make map pairs"""
import random
import numpy as np
import matplotlib.pyplot as plt
import Terrain.pathcheck
class Generator():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def generate_solveable_pair(self, x,y, octaves):
        """Generates a map pair that is solveable"""
        map1 = Terrain.pathcheck.path(x,y,octaves=octaves)
