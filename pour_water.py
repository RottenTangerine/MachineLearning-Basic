def operation(x, y, x_max, y_max):
    return {
        (x_max, y): 'fill x',
        (x, y_max): 'fill y',
        (0, y): 'empty x',
        (x, 0): 'empty y',
        (max(x - (y_max - y), 0), min((x + y), y_max)): 'x==>y',
        (min((x + y), x_max), max(y - (x_max - x), 0)): 'y==>x'
    }


def find_solution(x_max, y_max, target, status=(0, 0)):
    """
    This function tris to get certain amount of liquid using two cups whose volumes we already know.
    :param x_max: Cup A max volume
    :param y_max: Cup B max volume
    :param target: Goal
    :param status: Initial status of those two cups
    :return: Solution
    """
    if target in status:
        return status

    all_path = [[('init', status)]]
    all_status = set()

    while all_path:
        current_path = all_path.pop(0)
        current_statue = current_path[-1][-1]

        for k, v in (operation(*current_statue, x_max, y_max)).items():
            if k in all_status:
                continue
            all_status.add(k)
            current_path.append((v, k))
            if target in k:
                return current_path
            all_path.append(current_path.copy())
            current_path.pop()

    return 'No solution'


if __name__ == '__main__':
    def reformat_output(solution):
        for i, j in enumerate(solution):
            print(f'Step {i}: \t{j[0]}\t {j[1]}')
        return solution

    solution = find_solution(3, 7, 5)
    print(solution)
    reformat_output(solution)
