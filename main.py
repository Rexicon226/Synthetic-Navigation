import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import copy

# define the noise
noise = PerlinNoise(octaves=10, seed=1)
# define the size
xpix, ypix = 100, 100
threshold_amount = 0.5  # variable to round to (above this is black, below is white)


def threshold(value: float, amt: float) -> int:
    """converts the value to a 0 if it is less than amt, or 1 if it is greater than or equal to amt"""
    return int(value >= amt)


def thresholdedNoise(x: float, y: float, noise_func: PerlinNoise) -> int:
    """converts the perlinNoise output to a threshold (0 or 1)"""
    noise_here = noise_func([x, y])  # get the noise here. Value is in the interval [-1,1]
    noise_here += 1  # make the interval [0,2]
    noise_here /= 2  # make the interval [0,1]
    return threshold(noise_here, threshold_amount)  # round the interval to make it a 0 or 1


pic = [[thresholdedNoise(i / xpix, j / ypix, noise) for j in range(xpix)] for i in range(ypix)]
borderpic = copy.deepcopy(pic)  # create a deep copy so that stuff doesn't get messed up
for x in range(len(pic)):
    for y in range(len(pic)):
        if pic[x][y] == 1:
            mx = 0
            mn = 0
            if y < len(pic) - 1:
                mx += 1
                mn += pic[x][y + 1]
            if y > 0:
                mx += 1
                mn += pic[x][y - 1]
            if x < len(pic) - 1:
                mx += 1
                mn += pic[x + 1][y]
            if x > 0:
                mx += 1
                mn += pic[x - 1][y]

            if mx != mn:
                borderpic[x][y] = 1
            else:
                borderpic[x][y] = 0



        else:
            pic[x][y] = 0

            # borderpic[x][y] = mx-1
print(pic)
print(borderpic)
plt.imshow(pic, cmap='gray')
plt.imshow(borderpic, cmap='gray')
plt.show()
