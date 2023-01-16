import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import PriorityQueue
import time
import random

from CompositeEnvironment import Visualizer, Environment
from Terrain import pathcheck, generator


class AStarPathfinder:
    def __init__(self, terrain):
        self.terrain = terrain
        self.start = (0, 0)
        self.end = (31, 31)
        self.visited = set()
        self.path = []
        self.fig, self.ax = plt.subplots()

    @staticmethod
    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self):
        queue = PriorityQueue()
        queue.put((0, self.start))
        while not queue.empty():
            current = queue.get()[1]
            self.visited.add(current)

            if queue.qsize() != 0:
                print('Visited: {} - Queue: {}, {:.4f}%'.format(len(self.visited), queue.qsize(),
                                                                (len(self.visited) / queue.qsize())))

            if current == self.end:
                break

            neighbors = [(current[0] + x, current[1] + y) for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
            for neighbor in neighbors:
                if (0 <= neighbor[0] < 32) and (0 <= neighbor[1] < 32):
                    if self.terrain[neighbor[0], neighbor[1]] == 0 and neighbor not in self.visited:
                        self.path.append(neighbor)
                        queue.put((self.heuristic(self.end, neighbor) + self.heuristic(neighbor, self.start), neighbor))

    def animate(self):
        im = self.ax.imshow(self.terrain, animated=True)

        def updatefig(*args):
            if self.path:
                x, y = self.path.pop(0)
                im.set_array(self.terrain)
                self.ax.scatter(y, x, c='r', marker='x')
                time.sleep(0.1)
            else:
                im.set_array(self.terrain)
                self.ax.scatter(self.end[1], self.end[0], c='g', marker='x')
                return [im]

        ani = animation.FuncAnimation(self.fig, updatefig, interval=1)
        plt.show()


if __name__ == '__main__':
    x = random.randint(50, 200)
    y = random.randint(50, 200)

    noise_level = 30

    print("({}, {})".format(x, y))

    pic, seed = pathcheck.path(32, 32, 2)
    noisy_pic = generator.generateNoise(256, 256, 4, noise_level, seed, True)

    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))

    masked = ev.generate()

    vi = Visualizer('../ML/models/synthnav-model-0.pth', noisy_pic)

    # de_noised_original, loss = vi.dNoise(masked)
    # de_noised = vi.thresholdDNoise(de_noised_original, 0.5)

    # print("Processed Model Loss: {}".format(loss))

    print("Path Finding Start")
    pathfinder = AStarPathfinder(pic)
    pathfinder.a_star()
    pathfinder.animate()

    print("done")
