import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

import random

from tqdm import tqdm

from Terrain.timers import FunctionTimer
from CompositeEnvironment import Visualizer, Environment
from Terrain import pathcheck, generator


class AStarPathfinder:
    def __init__(self, terrain):
        self.terrain = terrain
        self.start = (0, 0)
        self.end = (255, 255)
        self.visited = set()
        self.path = []
        self.fig, self.ax = plt.subplots()

    @staticmethod
    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def a_star(self):
        queue = []
        heapq.heappush(queue, (self.heuristic(self.start, self.end), self.start))
        # Distance threshold
        max_distance = 256*256
        visited = set()
        visited.add(self.start)
        with tqdm(total=256*256) as pbar:
            while queue:
                current_distance, current = heapq.heappop(queue)

                if current_distance > max_distance:
                    break

                pbar.update(1)

                if current == self.end:
                    break

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x, y = current
                    neighbor = (x + dx, y + dy)

                    if (0 <= neighbor[0] < 256) and (0 <= neighbor[1] < 256):
                        if self.terrain[neighbor[0], neighbor[1]] == 0 and neighbor not in visited:
                            distance = self.heuristic(neighbor, self.end)
                            heapq.heappush(queue, (distance, neighbor))
                            self.path.append(neighbor)
                            visited.add(neighbor)

    def animate(self):
        im = self.ax.imshow(self.terrain, animated=True)

        def updatefig(*args):
            if self.path:
                x, y = self.path.pop(0)
                im.set_array(self.terrain)
                self.ax.scatter(y, x, c='r', marker='x')

            else:
                im.set_array(self.terrain)
                self.ax.scatter(self.end[1], self.end[0], c='g', marker='x')
                return [im]

        ani = animation.FuncAnimation(self.fig, updatefig, interval=1)
        ani.save('astar.mp4', writer='imagemagick', fps=60)
        plt.show()


if __name__ == '__main__':
    x = random.randint(50, 200)
    y = random.randint(50, 200)
    noise_level = 30

    print("({}, {})".format(x, y))

    pic, seed = pathcheck.path(256, 256, 5)
    noisy_pic = generator.generateNoise(256, 256, 5, noise_level, seed, True)

    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))

    masked = ev.generate()

    vi = Visualizer('../ML/models/synthnav-model-0.pth', noisy_pic)

    de_noised_original, loss = vi.dNoise(masked)
    de_noised = vi.thresholdDNoise(de_noised_original, 0.5)

    print("Processed Model Loss: {:.4}".format(loss))

    pathfinder = AStarPathfinder(de_noised)
    f = FunctionTimer("Path Finding")
    pathfinder.a_star()
    f.stop()
    pathfinder.animate()

    print("done")
