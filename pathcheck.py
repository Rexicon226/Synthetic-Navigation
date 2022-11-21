import terraingen
import random
import matplotlib.pyplot as plt
row = 5
col = 3

def isPath(arr):
    arr[0][0] = 1

    for i in range(1, row):
        if (arr[i][0] != -1):
            arr[i][0] = arr[i - 1][0]

    for j in range(1, col):
        if (arr[0][j] != -1):
            arr[0][j] = arr[0][j - 1]

    for i in range(1, row):
        for j in range(1, col):
            if (arr[i][j] != -1):
                arr[i][j] = max(arr[i][j - 1],
                                arr[i - 1][j])

    return (arr[row - 1][col - 1] == 1)
solved = False
failedSeeds = []
while(solved == False):
    seed = random.randint(1, 1000)
    if failedSeeds.count(seed) == 0:
        solved = isPath(terraingen.terrain(10, seed))
        if solved == True:
            print(seed)
            print(len(failedSeeds))
            print(failedSeeds)
            plt.imshow(terraingen.terrain(10, seed), cmap='gray')
            plt.show()
        print(f'Failed: ' + str(seed))
        failedSeeds.append(seed)
