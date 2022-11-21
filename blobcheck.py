import pathcheck
import terraingen
import matplotlib.pyplot as plt

pic = pathcheck.path(15, 3)
plt.imshow(pic, cmap='gray')
plt.show()