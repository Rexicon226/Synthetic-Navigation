import time


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

    def __init__(self):
        self.startTime = time.perf_counter()

    @property
    def time(self):
        return time.perf_counter() - self.startTime


class FunctionTimer(BaseTimer):
    def __init__(self, name):
        self.name = name
        print(f"----- starting \"{self.name}\" -----")
        super().__init__()

    def stop(self):
        ts = self.time
        print(f"----- done. \"{self.name}\" took {ts}s -----")


if __name__ == "__main__":
    x = FunctionTimer("testing")
    time.sleep(2)
    x.stop()
