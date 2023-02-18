"""Generator used to make map pairs"""
import random
import numpy as np
import matplotlib.pyplot as plt
from Terrain.map import Map
import asyncio
class Generator():
    def __init__(self, width, height, octaves=1):
        self.width = width
        self.height = height
        self.octaves = octaves

    def generate_solveable_pair(self ):
        """Generates a map pair that is solveable"""
        real = Map.solveable_map(self.width, self.height, self.octaves)
        predicted = real + Map.built(self.width, self.height, self.octaves)


