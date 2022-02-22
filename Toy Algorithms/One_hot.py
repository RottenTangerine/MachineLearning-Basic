import numpy as np


def one_hot(elements):
    pure = list(set(elements))
    return np.diag(np.ones(len(pure)))


data = ['北京', '上海', '成都', '重庆', '北京', '广州', '澳门']
print(one_hot(data))
