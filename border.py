import copy

import matplotlib.pyplot as plt

import pathcheck

pic = pathcheck.path(100, 5, 375584)

borderpic = copy.deepcopy(pic)  # create a deep copy so that stuff doesn't get messed up

for i in range(len(borderpic)):
    borderpic[i] = [abs(ele) for ele in borderpic[i]]

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
print("Pic: " + str(pic))
print("Border: " + str(borderpic))
plt.imshow(pic, cmap='winter', alpha=1)
plt.imshow(borderpic, cmap='binary', alpha=0.8)
plt.show()
