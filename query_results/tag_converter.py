import json


class Tagger:
    def __init__(self):
        self.untagged = {}
        self.tagged = {}
    
    def load_untagged(self, file):
        self.untagged = load_array(file)
    
    def tag(self, id, tag):
    


def dump_array(file, array, overwrite=True):
    if type(array) is not list:
        raise TypeError("Dict array not of valid type.")
    with open(file, 'w' if overwrite else 'a') as fp:
        for element in array:
            fp.write(json.dumps(element) + '\n')


def load_array(file):
    array = []
    with open(file, 'r') as fp:
        for line in fp.readlines():
            array.append(json.loads(line.strip()))
    return array


