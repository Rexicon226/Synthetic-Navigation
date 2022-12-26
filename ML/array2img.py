import random

import PIL
import numpy as np
from tqdm import tqdm
import threading
import generator
import math


def array2image(x, y, octaves, weight, seed: int = 0, iD: int = 0):
    array = generator.generateClean(x, y, octaves, seed)
    bool_array = np.array(array, dtype=bool)
    img = PIL.Image.fromarray(bool_array)
    img.save('/home/dr/Synthetic-Navigation/ML/train_images/clean/' + str(iD) + '_image.png', bits=1, optimize=True)
    array = generator.generateNoise(x, y, octaves, weight, seed)
    bool_array = np.array(array, dtype=bool)
    img = PIL.Image.fromarray(bool_array)
    img.save('/home/dr/Synthetic-Navigation/ML/train_images/noise/' + str(iD) + '_image.png', bits=1, optimize=True)


image_count = 20000
threads = 64
size = 256
octaves = 5
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


    # No purpose other than wait main thread until end for print
    for x in thread_array:
        x.join()


generate()
print("done")
