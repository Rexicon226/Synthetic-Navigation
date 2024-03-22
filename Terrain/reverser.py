import Terrain.terraingen as terraingen


def reverse(pic: list):
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            if pic[i][j] == 0:
                pic[i][j] = -1
            else:
                pic[i][j] = 0
    return pic


if __name__ == "__main__":
    pic = terraingen.terrain(10, 10, 3, True, 12309)
    print(pic)
    pic = reverse(pic)
    print(pic)
