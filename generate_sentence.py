from icecream import ic
import random
import re

grammar = '''
simple_sentence = subject verb predicate
complex_sentence = simple_sentence , conjunction complex_sentence/ simple_sentence .
subject = I/ you/ they
verb = play/ have/ like
predicate = tangerine/ football/ bicycle
conjunction = and/ moreover/ furthermore
'''


def get_materials(grammar_string: str):
    grammar_gen = dict()
    grammar_lists = grammar_string.split('\n')
    for line in grammar_lists:
        if not line.strip():
            continue
        statement, expression = line.split('=')
        expression = expression.split('/')
        grammar_gen[statement.strip()] = [i.strip() for i in expression]

    def generate_sentence(gram, target='complex_sentence'):
        if target not in gram:
            return target
        return ' '.join([generate_sentence(gram, i) for i in random.choice(gram[target]).split()])

    def reformat(sentence: str):
        sentence = sentence[:1].upper() + sentence[1:]
        pattern = r'.[,.]'
        return re.sub(pattern, lambda a: a.group()[1], sentence)

    return reformat(generate_sentence(grammar_gen))


if __name__ == '__main__':
    print(get_materials(grammar))
