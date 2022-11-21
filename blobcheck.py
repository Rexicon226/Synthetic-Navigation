import matplotlib.pyplot as plt
import border
import copy


def solve(matrix):
    """TODO: @Rexicon266 add comments and docstring"""
    def dfs(i, j):
        if i < 0 or j < 0 or i >= R or j >= C:
            return True
        if matrix[i][j] == 0:
            return True
        matrix[i][j] = 0
        a = dfs(i + 1, j)
        b = dfs(i - 1, j)
        c = dfs(i, j + 1)
        d = dfs(i, j - 1)
        return a and b and c and d

    R, C = len(matrix), len(matrix[0])
    ans = 0
    for i in range(R):
        for j in range(C):
            if matrix[i][j] == 1:
                if dfs(i, j):
                    ans += 1
    return ans


def blobs(pic: list[list[int]]):
    """Calculates the number of "blobs" in the given array

    Parameters
    ----------
    pic : list[list[1,0]]
        2 dimensional binary array of the "terrain". Typically generated with perlin noise.

    Returns
    -------
    islands, oceans : int,int
        number of regions with a value of 0, number of regions with a value of 1"""
    islandpic = copy.deepcopy(pic)
    oceanpic = copy.deepcopy(pic)
    for y in range(len(oceanpic) - 1):
        oceanpic[y] = [0 if (k == 1) else 1 for k in oceanpic[y]]

    islands = solve(islandpic)
    oceans = solve(oceanpic)

    return islands, oceans


if __name__ == "__main__":
    from terraingen import terrain

    pic = terrain(5, 1, 50, 50)
    borderpic = border.bordercheck(pic)
    islands, oceans = blobs(pic)

    print("Islands: " + str(islands))
    print("Oceans: " + str(oceans))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    axes[0][0].imshow(pic, cmap='winter_r')
    axes[0][1].imshow(borderpic, cmap='binary')
    axes[1][0].imshow(pic, cmap='winter_r')
    axes[1][0].imshow(borderpic, cmap='binary', alpha=0.8)
    plt.show()
