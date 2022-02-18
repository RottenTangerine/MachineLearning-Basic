import numpy as np
from icecream import ic
import math
import random
from matplotlib import pyplot as plt

locations = {
    '甘肃': [103.73, 36.03],
    '青海': [101.74, 36.56],
    '四川': [104.06, 30.67],
    '河北': [114.48, 38.03],
    '云南': [102.73, 25.04],
    '贵州': [106.71, 26.57],
    '湖北': [114.31, 30.52],
    '河南': [113.65, 34.76],
    '山东': [117, 36.65],
    '江苏': [118.78, 32.04],
    '安徽': [117.27, 31.86],
    '浙江': [120.19, 30.26],
    '江西': [115.89, 28.68],
    '福建': [119.3, 26.08],
    '广东': [113.23, 23.16],
    '湖南': [113, 28.21],
    '海南': [110.35, 20.02],
    '辽宁': [123.38, 41.8],
    '吉林': [125.35, 43.88],
    '黑龙江': [126.63, 45.75],
    '山西': [112.53, 37.87],
    '陕西': [108.95, 34.27],
    '台湾': [121.30, 25.03],
    '北京': [116.46, 39.92],
    '上海': [121.48, 31.22],
    '重庆': [106.54, 29.59],
    '天津': [117.2, 39.13],
    '内蒙古': [111.65, 40.82],
    '广西': [108.33, 22.84],
    '西藏': [91.11, 29.97],
    '宁夏': [106.27, 38.47],
    '新疆': [87.68, 43.77],
    '香港': [114.17, 22.28],
    '澳门': [113.54, 22.19]
}


def geo_distance(source, destination):
    lon1, lat1 = source
    lon2, lat2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def generate_random_center(x, y):
    x = random.uniform(min(x), max(x))
    y = random.uniform(min(y), max(y))
    return x, y


k = 5

all_location = locations.values()
all_x = [i[0] for i in all_location]
all_y = [i[1] for i in all_location]
ic(all_x, all_y)

centers = np.asarray([generate_random_center(all_x, all_y) for i in range(k)])
ic(centers)

while True:
    centers_backup = centers.copy()
    location_list = []
    for location in all_location:
        location_list += [min([[np.asarray(location), index, geo_distance(location, center)]
                               for index, center in enumerate(centers)],
                              key=lambda a: a[2])[:2]]
    location_list = np.asarray(location_list)

    for i in range(k+1):
        if len(location_list[location_list[:, 1] == i]) != 0:
            centers[i] = np.mean(location_list[location_list[:, 1] == i][:, 0])

    ic(centers)
    fig = plt.figure()
    fig, axs = plt.subplots()
    axs.scatter(all_x, all_y)
    axs.scatter(centers[:, 0], centers[:, 1])
    plt.show()

    if np.equal(centers_backup, centers.copy()).all():
        break


