import os
import json


# class User:
#     def __init__(self, uid, log_base='./log'):
#         self.record = {}
#         self.uid = uid
#         self.log_base = log_base
#         self.log_file = append_slash_if_necessary(self.log_base) + self.uid + '.log'
#
#     def dump_record(self):
#         dump_array(self.log_file, self.record)
#
#     def load_record(self):
#         record = load_array(self.log_file)
#         if not record:
#             dump_array(self.log_file, [{}])
#             return [{}]
#         else:
#             return record


class Tagger:
    def __init__(self):
        self.theme = ''
        self.description = ''
        self.return_path = ''
        self.untagged = {}
        self.tagged = {}
    
    def load_untagged(self, file):
        # untagged = {id(int): 'text', ...}
        self.theme, self.description, self.return_path, self.untagged = load_array(file)
        return self.theme, self.description, self.return_path, self.untagged
    
    def dump_tagged(self, file):
        # self.tagged = {id1(int): {'ltimes': 123, 'pos': 23, 'neg':5} , ...}
        dump_array(file, [self.tagged])
    
    def tag(self, id_, tag):
        if tag not in [-1, 1]:
            return
        if id_ not in self.tagged:
            self.tagged[id_] = {'lbl': 1, 'pos': 1 if tag == 1 else 0, 'neg': 1 if tag == -1 else 0}
        else:
            self.tagged[id_]['lbl'] += 1
            self.tagged[id_]['pos'] += 1 if tag == 1 else 0
            self.tagged[id_]['neg'] += 1 if tag == -1 else 0


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


# def append_slash_if_necessary(path):
#     return path + os.path.sep if not path.endswith(os.path.sep) else ''
