"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""

from math import floor, log10


def round_sig(x, sig=2):
    """
    Rounds `x` to `sig` significant figures.
    """
    return round(x, sig - int(floor(log10(abs(x)))) - 1)
