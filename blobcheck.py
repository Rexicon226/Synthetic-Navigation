import matplotlib.pyplot as plt
import border

pic, borderpic = border.bordercheck(100, 5, 375584)

#insert smart code to detect amount of islands here :brain:

plt.imshow(pic, cmap='winter_r', alpha=1)
plt.imshow(borderpic, cmap='binary', alpha=0.8)
plt.show()

