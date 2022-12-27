import numpy as np

from Terrain import generator
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, image, noisy, radius):
        self.image = image.copy()
        self.radius = radius
        self.noisy_image = noisy.copy()

    def generate(self):
        masked = get_visible_image(self.image, self.radius, self.noisy_image)
        return masked


def create_circular_mask(h, w, radius, center=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_visible_image(image, radius, noisy):
    # Find the size of the image
    image = np.abs(image)

    h, w = image.shape[:2]
    mask = create_circular_mask(h, w, radius)
    masked_img = image.copy()
    mask = np.array(mask, dtype=int)
    print(mask)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 0:
                masked_img[i][j] = noisy[i][j]
            else:
                masked_img[i][j] = image[i][j]

    return noisy, masked_img


if __name__ == "__main__":
    seed = 868190740987676311
    pic = np.array(generator.generateClean(256, 256, 5, seed, True))
    noisy_pic = np.array(generator.generateNoise(256, 256, 5, 30, seed, True))
    vi = Environment(pic, noisy_pic, 16)

    noisy, masked = vi.generate()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(noisy)
    axs[1].imshow(masked)
    plt.show()
