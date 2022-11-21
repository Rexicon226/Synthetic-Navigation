import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

# define the noise
noise = PerlinNoise(octaves=10, seed=1)
# define the size
xpix, ypix = 100, 100
threshold_amount = 0.5 # variable to round to (above this is black, below is white)

def threshold(value: float, amt:float) -> int:
    """converts the value to a 0 if it is less than amt, or 1 if it is greater than or equal to amt"""
    return int(value >= amt)

def thresholdedNoise(x: float, y: float, noise_func: PerlinNoise) -> int:
    """converts the perlinNoise output to a threshold (0 or 1)"""
    noise_here = noise_func([x, y])  # get the noise here. Value is in the interval [-1,1]
    noise_here += 1  # make the interval [0,2]
    noise_here /= 2  # make the interval [0,1]
    return threshold(noise_here, threshold_amount)  # round the interval to make it a 0 or 1


pic = [[thresholdedNoise(i / xpix, j / ypix, noise) for j in range(xpix)] for i in range(ypix)]

plt.imshow(pic, cmap='gray')
plt.show()
