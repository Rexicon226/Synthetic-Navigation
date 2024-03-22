import time
from Terrain.helpers import round_sig


class BaseTimer:
    """class for timing things such as testing performance

    Usage
    -----
    Instantiate when timer begins
    property Time().time returns time elapsed

    Properties
    ----------
    time : float
        time since object was instantiated"""

    startTime: float

    def __init__(self):
        self.startTime = time.perf_counter()

    @property
    def time(self) -> float:
        """
        Returns the time elapsed since the timer was instantiated
        """
        return time.perf_counter() - self.startTime


class FunctionTimer(BaseTimer):
    name: str
    totalTime: float
    stopTime: float

    def __init__(self, name):
        self.name = name

    def start(self):
        """
        Starts the timer
        """
        print(f'----- Starting "{self.name}" -----')
        super().__init__()

    def stop(self):
        """
        Stops the timer and prints the time elapsed
        """
        self.stopTime = time.perf_counter()
        self.totalTime = self.stopTime - self.startTime

    def print(self):
        """
        Prints the time elapsed
        """
        print(f'----- "{self.name}" took {round_sig(self.totalTime)} seconds ----- ')


if __name__ == "__main__":
    x = FunctionTimer("testing")
    x.start()

    time.sleep(1)

    x.stop()
    x.print()
