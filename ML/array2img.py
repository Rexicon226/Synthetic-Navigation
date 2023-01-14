import random

import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm
import threading
from Terrain import generator
import math
from Terrain import timers
import os
import cv2


def array2image(x, y, octaves, weight, seed: int = 0, iD: int = 0):
    timer = timers.FunctionTimer("Clean - Generate")
    array = generator.generateClean(x, y, octaves, seed)
    timer.stop()
    bool_array = np.array(array, dtype=bool)
    timer = timers.FunctionTimer("Clean - To Image")
    img = PIL.Image.fromarray(bool_array)
    print(img)
    timer.stop()
    timer = timers.FunctionTimer("Clean - Image Save")
    img.save('./train_images/clean/' + str(iD) + '_clean.jpeg', bits=1, optimize=True)
    timer.stop()
    timer = timers.FunctionTimer("Noisy - Generate")
    array = generator.generateNoise(x, y, octaves, weight, seed)
    timer.stop()
    bool_array = np.array(array, dtype=bool)
    timer = timers.FunctionTimer("Noisy - To Image")
    img = PIL.Image.fromarray(bool_array)
    timer.stop()
    timer = timers.FunctionTimer("Noisy - Image Save")
    img.save('./train_images/noisy/' + str(iD) + '_noisy.jpeg', bits=1, optimize=True)
    timer.stop()


def convert_to_1bit(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith(".jpeg"):
            img = Image.open(os.path.join(input_dir, file))
            img_array = np.array(img)
            for i in range(len(img_array)):
                for j in range(len(img_array[i])):
                    if img_array[i][j] < 128:
                        img_array[i][j] = 0
                    elif img_array[i][j] > 128:
                        img_array[i][j] = 255
                    else:
                        print("Uncertain Image")
            bool_array = np.array(img_array, dtype=bool)
            img = PIL.Image.fromarray(bool_array)
            img.save('./train_images/noisy/' + file, bits=1, optimize=True)



threshold = 128
image_count = 1
threads = 1
size = 256
octaves = 4
weight = 30


def generate():
    current_count = 0

    array_count = current_count
    final_count = image_count + current_count

    splitting_array = [0] * threads

    thread_count = math.ceil(image_count / threads)

    print(thread_count)

    for i in range(threads):
        if (array_count + thread_count) <= final_count:
            array_count = array_count + thread_count
            splitting_array[i] = thread_count
        else:
            splitting_array[i] = final_count - array_count

    def thread(x):
        count = current_count + (splitting_array[x] * x)
        for i in range(splitting_array[x]):
            seed = random.randint(1, 1000000000000)
            array2image(size, size, octaves, weight, seed, count + i)

    thread_array = []

    for i in range(threads):
        x = threading.Thread(target=thread, args=(i,))
        thread_array.append(x)
        x.start()

    for x in thread_array:
        x.join()


convert_to_1bit("train_images/noisy/", "train_images/noisy/")
print("done")
