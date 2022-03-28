import random
import re
import rules

rule_pattern = re.compile(r'\(\.\*\)')

def flatten(nest):
    match nest:
        case []: return []
        case list() | tuple() | set():
            return flatten(nest[0]) + flatten(nest[1:])
        case _: return [nest]

def match_rule(_sentence, _rules):
    for _rule in _rules:
        string = split_by_rule(_sentence, _rule[0])
        case = generate_match_case(_rule[0])
        response = generate_response(_rule)
        script = f"""def _match(string):
            match string:
                case {case}:
                    print(f"- {response}")
                    
                    return True
                case _:
                    return False
        """

        exec (script)
        match = eval(f'_match(string)')
        if match:
            break

def generate_match_case(_rule):
    rule_list = split_by_rule(_rule, _rule)
    keyword_counter = 1
    for i in range(len(rule_list)):
        if rule_pattern.match(rule_list[i]):
            rule_list[i] = f'x{keyword_counter}'
            keyword_counter += 1
    rule_list = re.sub("'(x[0-9])'", r'\1', str(rule_list))
    return rule_list

def split_by_rule(_sentence, _rule):
    key_word = [s.strip() for s in rule_pattern.sub(' ', _rule).split()]
    return split_sentence(_sentence, key_word) if key_word else [_sentence]

def split_sentence(_sentence, _kw):
    pattern = re.compile(r'|'.join(_kw))
    index_list = [s.span() for s in pattern.finditer(_sentence)]
    index_list = flatten(index_list)
    if index_list:
        if index_list[0] != 0:
            index_list = [0] + index_list
        if index_list[-1] != len(_sentence):
            index_list.append(len(_sentence))
    sentence_list = [_sentence[j: index_list[i + 1]].strip() for i, j in enumerate(index_list[:-1])]
    return sentence_list

def generate_response(_rule):
    res = random.choice(_rule[1])
    res_list = re.sub('%([0-9]+)',lambda x: f'{{x{x.group(1)}}}', res)
    return res_list


if __name__ == '__main__':
    while True:
        # sentence = 'I need to do some exercise'
        sentence = input()
        rule = rules.pairs
        match_rule(sentence, rule)
