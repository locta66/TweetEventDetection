import os
import time
import json
from json import JSONDecodeError
import bz2file


def slash_appender(func):
    def decorator(*args, **kwargs):
        string = func(*args, **kwargs)
        string = string + os.path.sep if not string.endswith(os.path.sep) else string
        return string
    return decorator


def sync_real_time_counter(info):
    def time_counter(func):
        def decorator(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            print('function name:', func.__name__, ',', info, 'time elapsed:', time.time() - start_time, 's')
        return decorator
    return time_counter


def write_lines(file, lines, mode='w'):
    with open(file, mode) as fp:
        fp.writelines(lines)


def dump_array(file, array, overwrite=True, sort_keys=False):
    if type(array) is not list:
        raise TypeError("Input should be a list.")
    lines = list()
    for element in array:
        lines.append(json.dumps(element, sort_keys=sort_keys) + '\n')
    with open(file, 'w' if overwrite else 'a') as fp:
        fp.writelines(lines)


def load_array(file):
    array = list()
    with open(file, 'r') as fp:
        for line in fp.readlines():
            array.append(json.loads(line.strip()))
    return array


def load_array_catch(file):
    array = list()
    with open(file, 'r') as fp:
        for idx, line in enumerate(fp.readlines()):
            try:
                array.append(json.loads(line.strip()))
            except JSONDecodeError as e:
                print('file:{}, line:{}, col:{}'.format(file,  e.lineno, e.colno, ))
                continue
    return array


def load_twarr_from_bz2(bz2_file):
    fp = bz2file.open(bz2_file, 'r')
    twarr = list()
    for line in fp.readlines():
        try:
            json_obj = json.loads(line.decode('utf8'))
            twarr.append(json_obj)
        except:
            print('Error when parsing ' + bz2_file + ': ' + line)
            continue
    fp.close()
    return twarr
