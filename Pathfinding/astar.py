import numpy as np
import matplotlib.pyplot as plt
import heapq

import random

from Terrain.timer import FunctionTimer
from CompositeEnvironment import Visualizer, Environment
from Terrain import pathcheck, generator


class AStarPathfinder:
    def __init__(self, terrain):
        self.terrain = terrain
        self.start = (0, 0)
        self.end = (255, 255)
        self.visited = set()
        self.path = []
        self.fig, self.ax = plt.subplots(2, 2)

    @staticmethod
    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self):
        queue = []
        heapq.heappush(queue, (self.heuristic(self.start, self.end), self.start))
        # Distance threshold
        max_distance = 256 * 256
        visited = set()
        visited.add(self.start)

        while queue:
            current_distance, current = heapq.heappop(queue)

            if current_distance > max_distance:
                break

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
        return self.path

    @staticmethod
    def overlay_path(path_matrix, terrain_matrix):
        overlay_matrix = np.maximum(path_matrix, terrain_matrix)
        return overlay_matrix

    @staticmethod
    def path2matrix(path):
        matrix = np.zeros((256, 256))
        for x, y in path:
            matrix[x][y] = 2
        return matrix

    def animate(self):
        self.ax[1][0].imshow(self.terrain, cmap="plasma")
        path_matrix = self.path2matrix(path)
        # overlay_matrix = self.overlay_path(path_matrix, self.terrain)
        self.ax[1][0].imshow(path_matrix, cmap="Reds")

        self.ax[0][1].imshow(masked)
        self.ax[1][0].set_title("Path")
        self.ax[0][1].set_title("Input")

        self.ax[1][1].hist(de_noised_original, bins=25)
        self.ax[1][1].set_title("De-Noised Image Histogram")

        self.ax[0][0].imshow(de_noised_original, cmap="plasma")
        self.ax[0][0].set_title("De-Noised Image")

        self.fig.suptitle(
            "A* Pathfinding Example" "\nImage Size: 256 x 256\n" "Noise Level: {}%\nAccuracy: {:.2f}%".format(
                noise_level, loss
            ),
            fontsize=16,
        )

        self.fig.set_size_inches(18.5, 10.5)
        plt.show()


if __name__ == "__main__":
    x = random.randint(50, 200)
    y = random.randint(50, 200)
    noise_level = 30

    print("({}, {})".format(x, y))

    pic, seed = pathcheck.path(256, 256, 5)
    noisy_pic = generator.generateNoise(256, 256, 5, noise_level, seed, True)

    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))

    masked = ev.generate()

    vi = Visualizer("../DNoise/models/synthnav-model-0.pth", noisy_pic)

    de_noised_original, loss = vi.dNoise(masked)
    de_noised = vi.thresholdDNoise(de_noised_original, 0.5)

    print("Processed Model Loss: {:.4}".format(loss))

    pathfinder = AStarPathfinder(de_noised)
    f = FunctionTimer("Path Finding")
    path = pathfinder.a_star()
    f.stop()
    pathfinder.animate()

    print("done")
