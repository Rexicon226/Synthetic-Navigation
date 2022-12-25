from Terrain import generator
from Terrain import reverser
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2
import PIL
from skimage import data, io, util


def array2image(x, y, octaves, weight, seed: int = 0, iD: int = 0):
    array = generator.generateClean(x, y, octaves, seed)
    bool_array = np.array(array, dtype=bool)
    img = PIL.Image.fromarray(bool_array)
    img.save('./train_images/clean/' + str(iD) + '_image.png', bits=1, optimize=True)
    array = generator.generateNoise(x, y, octaves, weight, seed)
    bool_array = np.array(array, dtype=bool)
    img = PIL.Image.fromarray(bool_array)
    img.save('./train_images/noise/' + str(iD) + '_image.png', bits=1, optimize=True)


image_count = 1000


def greyscale():
    for i in tqdm(range(image_count)):
        image = cv2.imread('./images/noise/inputs/' + str(i) + '_image.png')

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        plt.imsave('images/noise/inputs/' + str(i) + '_image.png', gray_image, cmap='gray', bits=1)


def generate():
    for i in tqdm(range(image_count)):
        seed = random.randint(1, 10000000000)
        array2image(50, 50, 4, 10, seed, i)


generate()
print("done")
