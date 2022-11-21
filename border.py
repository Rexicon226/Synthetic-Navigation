import copy
import time
import pathcheck
import numpy as np

def bordercheck(size, octaves, seed=0):
    print('Starting Border Find')
    start_time = time.time()
    pic = pathcheck.path(size, octaves, seed)

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
    print("%s seconds of processing" % np.round(time.time() - start_time, 2))
    print("---Done Border Check---")
    return pic, borderpic
