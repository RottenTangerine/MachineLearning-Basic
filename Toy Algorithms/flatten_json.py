# python==3.10 patten match statement
# if we write in if statement, the code will be very complicated and hard to handle nested data
def parse_json(json_string):
    match json_string:
        case {'text': str(name), 'color': str(c)}:
            print(f'Give {name} with color {c}')
        case {'sleep': str(state), 'time': int(t), 'person': {'name': str(name), 'age': int(age)}}:
            print(f'{name} with age {age} has been {state} for {t}')
        case {'sleep': str(state), 'time': int(t)}:
            print(f'{state} for {t}')
        case _:
            raise TypeError('invalid json')

if __name__ == '__main__':
    color_json = {'text': 'Car', 'color': 'Red'}
    state_json = {'sleep': 'off', 'time': 10}
    state_with_person = {'sleep': 'off', 'time': 10, 'person': {'name': 'John', 'age': 30}}


    parse_json(color_json)
    parse_json(state_json)
    parse_json(state_with_person)
