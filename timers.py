import time


class Timer:
    """class for timing things such as testing performance

    Usage
    -----
    Instantiate when timer begins
    property Time().time returns time elapsed

    Properties
    ----------
    time : float
        time since object was instantiated"""

    def __init__(self):
        self.startTime = time.perf_counter()

    @property
    def time(self):
        return time.perf_counter() - self.startTime
