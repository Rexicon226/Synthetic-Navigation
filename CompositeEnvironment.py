import random

import numpy as np

from Terrain import generator
import matplotlib.pyplot as plt
from DNoise.dnoise import EncoderDecoder as ed
from Terrain.timing import FunctionTimer
import torch
from torch import nn

Image = np.ndarray


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


loss_fn = MAELoss()


class Environment:
    def __init__(self, image: Image, noisy: Image, radius: int, center: None | tuple) -> None:
        self.image = image.copy()
        self.radius = radius
        self.noisy_image = noisy.copy()
        self.center = center

    def generate(self) -> Image:
        masked = get_visible_image(self.image, self.radius, self.noisy_image, self.center)
        return masked


def create_circular_mask(h: int, w: int, radius: int, center: None | tuple = None) -> np.ndarray:
    if center is None:  # use the middle of the image
        center = (w / 2, h / 2)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_visible_image(image: Image, radius: int, noisy: Image, center: None | tuple) -> Image:
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
    def __init__(self, model_path, original):
        self.model_path = model_path
        self.original = original.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dNoiseVis(self, inputpic):
        print("Loaded Model")
        model = ed().to(self.device)
        model.eval()
        model.load_state_dict(torch.load(f=self.model_path))

        inputpic = np.array(inputpic, dtype=float)
        inputpic = torch.tensor(inputpic, dtype=torch.float32).view(1, 1, 256, 256)
        inputpic = inputpic.type(torch.cuda.FloatTensor)

        de_noise_timer = FunctionTimer("De-Noising")
        de_noise_timer.start()

        de_noised_image = model(inputpic)

        de_noise_timer.stop()
        de_noise_timer.print()

        loss = loss_fn(de_noised_image, inputpic)

        loss = (1 - loss.item()) * 100

        de_noised_image = de_noised_image.view(256, 256)

        inputpic = inputpic.view(256, 256).cpu()

        de_noised_image = de_noised_image.detach()
        de_noised_image = de_noised_image.cpu().numpy()
        print("Processed Image")
        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(de_noised_image, cmap="plasma_r")
        ax[0][0].set_title("De-Noised Image")
        ax[0][1].imshow(inputpic, cmap="plasma_r")
        ax[0][1].set_title("Noisy Image")
        ax[1][0].imshow(self.original, cmap="plasma_r")
        ax[1][0].set_title("Ground Image")
        ax[1][1].hist(de_noised_image, bins=25)
        ax[1][1].set_title("De-Noised Image Histogram")

        fig.suptitle(
            "Image Size: 256 x 256\nNoise Level: {}%\nAccuracy: {:.2f}%".format(noise_level, loss),
            fontsize=16,
            y=0.9,
        )
        plt.show()

    def dNoise(self, image):
        print("Loaded Model")
        model = ed().to(self.device)
        model.eval()
        model.load_state_dict(torch.load(f=self.model_path))

        image = np.array(image, dtype=float)
        image = torch.tensor(image, dtype=torch.float32).view(1, 1, 256, 256)
        image = image.type(torch.cuda.FloatTensor)

        de_noised_image = model(image)
        loss = loss_fn(de_noised_image, image)

        loss = (1 - loss.item()) * 100

        de_noised_image = de_noised_image.view(256, 256)

        de_noised_image = de_noised_image.detach()
        de_noised_image = de_noised_image.cpu().numpy()

        return de_noised_image, loss

    @staticmethod
    def thresholdDNoise(input, x):
        output_image = np.clip(input, 0, 1)
        output_image[output_image < x] = 0
        output_image[output_image >= x] = 1
        return output_image


if __name__ == "__main__":
    seed = random.randint(1, 100000000000)
    x = random.randint(50, 200)
    y = random.randint(50, 200)
    noise_level = 80
    print("({}, {})".format(x, y))
    pic = np.array(generator.generateClean(256, 256, 5, seed, True))
    noisy_pic = np.array(generator.generateNoise(256, 256, 5, noise_level, seed, True))
    pic, noisy_pic = np.abs(pic), np.abs(noisy_pic)

    ev = Environment(pic, noisy_pic, 50, center=(x, y))
    masked = ev.generate()

    vi = Visualizer("./DNoise/models/synthnav-model-0.pth", pic)
    vi.dNoiseVis(masked)
