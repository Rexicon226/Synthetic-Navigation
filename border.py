import copy

import matplotlib.pyplot as plt

import pathcheck

pic = pathcheck.path(40, 4, 490738)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axes[0].imshow(pic, cmap='gray_r')
borderpic = copy.deepcopy(pic)  # create a deep copy so that stuff doesn't get messed up

for i in range(len(borderpic)):
    borderpic[i] = [abs(ele) for ele in borderpic[i]]
print("Pic Before: " + str(pic))
print("Border Before: " + str(borderpic))

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
            borderpic[x][y] = 0

            # borderpic[x][y] = mx-1
print("Pic After: " + str(pic))
print("Border After: " + str(borderpic))
axes[1].imshow(borderpic, cmap='Blues')
plt.show()
