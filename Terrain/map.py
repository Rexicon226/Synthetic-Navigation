"""Map class"""

import random

import numpy as np
import terraingen
import blobcheck
from Terrain import border, pathcheck


class Map:
    """Map class"""

    def __init__(self, width, height, threshold=0.5):
        self.width = width
        self.height = height
        self.map = np.array([[0 for x in range(width)] for y in range(height)])
        self.threshold = threshold

    @classmethod
    def from_array(cls, array):
        """Create a map from a given array"""
        map = cls(len(array[0]), len(array))
        map.map = array
        return map

    @classmethod
    def from_file(cls, filename):
        """Create a map from a given file"""
        with open(filename, "r") as file:
            lines = file.readlines()
            map = cls(len(lines[0].split(",")), len(lines))
            for y, line in enumerate(lines):
                for x, value in enumerate(line.split(",")):
                    map.set(x, y, int(value))
            return map

    @property
    def thresholded(self):
        """Get the thresholded map"""
        return np.array(
            [[1 if self.map[y][x] > self.threshold else 0 for x in range(self.width)] for y in range(self.height)]
        )

    @property
    def negative(self):
        """Get the negative of the map"""
        return Map.from_array((self.map * -1) + 1)

    @property
    def thresholdedNegative(self):
        """Get the thresholded negative of the map"""
        return self.negative.thresholded

    @classmethod
    def solveable_map(cls, width, height, octaves):
        """Generate a solveable map"""
        return cls.from_array(pathcheck.path(width, height, octaves=octaves))

    @classmethod
    def built(cls, width, height, octaves=1):
        """returns a built Map object"""
        map = cls(width, height)
        map.build(octaves)
        return map

    def __neg__(self):
        """Get the negative of the map"""
        return self.negative

    def build(self, octaves):
        """build the map"""
        self.map = terraingen.terrain(self.width, self.height, octaves, seed=random.randint(0, 1000000))

    def __mul__(self, other):
        """Multiply the map by a given value"""
        if isinstance(other, int) or isinstance(other, float):
            return self.map * other
        else:
            raise TypeError("Map object Can only be multiplied by int or float")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        """Add two maps together"""
        if isinstance(other, Map):
            return self.map + other.map
        else:
            raise TypeError("Map object Can only be added to another Map object")

    def get(self, x, y):
        """Get the value at a given position"""
        return self.map[y][x]

    def set(self, x, y, value):
        """Set the value at a given position"""
        self.map[y][x] = value

    @property
    def blobs(self):
        """Returns the number of blobs in the map
        SeeAlso
        -------
        blobcheck.check
        """
        return blobcheck.blobs(self.thresholded)

    @property
    def border(self):
        """Returns a map with the borders highlighted
        SeeAlso
        -------
        border.bordercheck
        """
        return border.bordercheck(self.thresholded)
