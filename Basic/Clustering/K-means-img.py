import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from icecream import ic

img = Image.open('img.png')
img = np.array(img)

def k_means(img, k):
    # stretch
    w, h, c = img.shape
    img = np.reshape(img, (-1, 3))

    # generate centers
    centers = np.random.randint(0, 2**8, (k, 3))
    while True:
        # calculate distance
        distance = np.stack([np.sum(np.power(img - c, 2), axis=-1) for c in centers])
        index = np.argmin(distance, axis=0)

        # assign nearest center
        neighbours = [img[index == c] for c in range(k)]

        # cal new center
        new_centers = np.asarray([np.mean(c, axis=0).astype(int) if c.size != 0 else centers[i] for i, c in enumerate(neighbours)])
        # check if change
        if (new_centers == centers).all():
            break
        centers = new_centers

    return np.reshape(index, (w, h))

k = 5
index = k_means(img, k)

# assign color
c_map = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 0, 0]
]

for i in range(k):
    img[index == i] = c_map[i]

# visualize
plt.imshow(img)
plt.show()
