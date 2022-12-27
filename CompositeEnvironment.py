import random

import numpy as np

from Terrain import generator
import matplotlib.pyplot as plt
from ML.main import EncoderDecoder as ed
import torch
import torch.nn as nn
from Terrain import reverser


class Environment:
    def __init__(self, image, noisy, radius, center=None):
        self.image = image.copy()
        self.radius = radius
        self.noisy_image = noisy.copy()
        self.center = center

    def generate(self):
        masked = get_visible_image(self.image, self.radius, self.noisy_image, self.center)
        return masked


def create_circular_mask(h, w, radius, center=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_visible_image(image, radius, noisy, center):
    # Find the size of the image
    image = np.abs(image)

    h, w = image.shape[:2]
    mask = create_circular_mask(h, w, radius, center)
    masked_img = image.copy()
    mask = np.array(mask, dtype=int)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 0:
                masked_img[i][j] = noisy[i][j]
            else:
                masked_img[i][j] = image[i][j]

    return masked_img


class Visualizer:

    def __init__(self, model_path, image):

        image = np.array(image, dtype=float)

        self.model_path = model_path
        self.image = image.copy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: {}\n".format(self.device))

    def dNoise(self):
        print("Loaded Model")
        model = ed().to(self.device)
        model.load_state_dict(torch.load(f=self.model_path))
        de_noised_image = model(torch.tensor(self.image, dtype=torch.float32).view(1, 1, 256, 256)).view(256, 256)

        de_noised_image = de_noised_image.detach()
        de_noised_image = de_noised_image.cpu()

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(de_noised_image, cmap='plasma_r')
        ax[0].set_title('De-Noised Image')
        ax[1].imshow(self.image, cmap='plasma_r')
        ax[1].set_title('Original')
        plt.show()


if __name__ == "__main__":
    seed = random.randint(1, 100000000000)
    x = random.randint(50, 200)
    y = random.randint(50, 200)
    print("({}, {})".format(x, y))
    pic = np.array(generator.generateClean(256, 256, 5, seed, True))
    noisy_pic = np.array(generator.generateNoise(256, 256, 5, 30, seed, True))
    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))

    masked = ev.generate()

    vi = Visualizer('./ML/models/synthnav-model-0.pth', masked)

    vi.dNoise()
