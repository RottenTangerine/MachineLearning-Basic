import math
import numpy as np
import matplotlib.pyplot as plt

# Generate data
data_num = 20
gen_data = lambda x, y, n: np.stack((np.random.uniform(*x, n), np.random.uniform(*y, n)), axis=1)

data_0 = gen_data((-0.5, 0.5), (1, 2), data_num)
data_1 = gen_data((-2, -1), (0, 1), data_num)
data_2 = gen_data((1, 2), (0, 1), data_num)

data = np.concatenate((data_0, data_1, data_2))
np.random.shuffle(data)


# K-means
def calculate_distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def generate_random_centers(arr, n):
    i = range(n)
    x = np.random.uniform(min(arr[:, 0]), max(arr[:, 0]), n)
    y = np.random.uniform(min(arr[:, 1]), max(arr[:, 1]), n)
    return np.stack((i, x, y), axis=1)


def scatter(a, x, y, i):
    color_map = ['green', 'orange', 'red']
    a.scatter(x, y, color=color_map[i])
    return


k = 3
centers = generate_random_centers(data, k)
step_counter = 0
while True:
    step_counter += 1
    fig, ax = plt.subplots()
    centers_backup = centers.copy()
    location_list = [[*min([(c[0], *location, calculate_distance(c[1:], location))
                            for c in centers], key=lambda a: a[-1])[:-1]]
                     for location in data]
    location_list = np.asarray(location_list)
    for index in range(k):
        dots = location_list[location_list[:, 0] == index][:, 1:]

        scatter(ax, dots[:, 0], dots[:, 1], index)
        ax.scatter(*centers[index][1:], marker='x', c='black')
        ax.set_title(f'Step: {step_counter}')
        ax.axis('equal')
        if len(dots) != 0:
            centers[index] = [index, *np.mean(dots, axis=0)]
    plt.show()
    if np.equal(centers_backup, centers).all():
        break
