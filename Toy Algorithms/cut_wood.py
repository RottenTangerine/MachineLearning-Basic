import math
from collections import defaultdict
from functools import lru_cache


def cut_wood(price_dict, wood_length):
    """
    This function can calculate the highest price that a wood can be sold and give the solution
    :param price_dict: price dictionary for each length of wood
    :param wood_length: length of wood
    :return: the possible highest price can sell and a list indicates how to cut the wood to sell the highest price
    """
    all_solutions = {}

    @lru_cache(maxsize=2 ** 20)
    def best_price(length):
        """
        :param length: length of wood
        :return: possible highest price can sell
        """
        # divide into two parts
        situations = [(price_dict[length], length, 0)] + \
                     [(best_price(i) + best_price(length - i), length - i, i)
                      for i in range(1, math.floor(length / 2) + 1)]
        highest_price, *solution = max(situations, key=lambda x: x[0])
        all_solutions[length] = solution

        return highest_price

    @lru_cache(maxsize=2 ** 20)
    def get_basic_unit(length):
        """
        :param length: length of wood
        :return: list of shorter woods length
        """
        if all_solutions[length][1] == 0:
            return [all_solutions[length][0]]

        basic_units = []
        for i in all_solutions[length]:
            basic_units += get_basic_unit(i)
        return basic_units

    return best_price(wood_length), get_basic_unit(wood_length)


if __name__ == '__main__':
    price_list = [2, 5, 6, 9, 10, 17, 17, 20, 26, 30, 33]
    dictionary = defaultdict(int)
    for le, price in enumerate(price_list):
        dictionary[le + 1] = price
    print(dictionary)
    print(cut_wood(dictionary, 48))
