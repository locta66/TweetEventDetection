import os
import re

import utils.function_utils as fu


def has_root_word(word):
    return word in synset_dict


def get_root_word(word):
    if word in synset_dict:
        return synset_dict[synset_dict[word]]
    else:
        return word


def get_synset(word):
    if word in synset_list_dict:
        return synset_list_dict[word]
    else:
        return [word]


filename = os.path.split(os.path.realpath(__file__))[0] + os.path.sep + 'synset.json'
# with open(filename, 'r') as fp:
#     synset_dict = json.loads(fp.readlines()[0])
synset_dict = fu.load_array(filename)[0]
synset_list_dict = dict()
for _word in synset_dict.keys():
    if re.findall('group\d+', _word):
        continue
    root_word = get_root_word(_word)
    if root_word in synset_list_dict:
        synset_list_dict[root_word].append(_word)
    else:
        synset_list_dict[root_word] = [_word]


if __name__ == "__main__":
    import sys
    while True:
        print('input your query:')
        query = sys.stdin.readline().strip('\n')
        print('has_root_word:', has_root_word(query), 'root_word:', get_root_word(query))
        print('synset', get_synset(query))
