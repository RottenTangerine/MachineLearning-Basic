def flatten(elements):
    if not elements:
        return []
    if isinstance(elements[0], (list, tuple, set)):
        return flatten(elements[0]) + flatten(elements[1:])
    else:
        return [elements[0]] + flatten(elements[1:])


# more elegant way by using match statement
def flatten_elegant(elements):
    match elements:
        case []: return []
        case list() | tuple() | set() as first, *remains:
            return flatten_elegant(first) + flatten_elegant(remains)
        case _: return [elements[0]] + flatten_elegant(elements[1:])


if __name__ == '__main__':
    sample = [0, 1, (2, 3)]
    L = ['a', 'b', ['cc', 'dd', ['eee', 'fff']], 'g', 'h']
    nest = [((0, 1), (2, 3), 4, 5, ((6, 7, 8), ((9, 10), ((11, 12, ((13, 14), 15), 16), 17))))]

    print(flatten(sample))
    print(flatten(L))
    print(flatten(nest))
