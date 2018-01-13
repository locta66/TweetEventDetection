import os
import time
import json
from json import JSONDecodeError
import multiprocessing as mp
import math
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


METHOD_EXTEND = 'extend'
METHOD_APPEND = 'append'
_SUPPORTED_MERGE_METHODS = {METHOD_EXTEND, METHOD_APPEND}


def merge_list(array, method=METHOD_EXTEND):
    if method not in _SUPPORTED_MERGE_METHODS:
        raise ValueError('param method incorrect: {}'.format(method))
    res = list()
    for item in array:
        if method == METHOD_EXTEND:
            res.extend(item)
        elif method == METHOD_APPEND:
            res.append(item)
    return res


def split_multi_format(array, process_num):
    block_size = math.ceil(len(array) / process_num)
    formatted_array = list()
    for i in range(process_num):
        formatted_array.append(array[i * block_size: (i + 1) * block_size])
    return formatted_array


def multi_process(func, args_list=None, kwargs_list=None):
    """
    Do func in multiprocess way.
    :param func: To be executed within every process
    :param args_list: default () as param for apply_async if not given
    :param kwargs_list:
    :return:
    """
    process_num = len(args_list)
    pool = mp.Pool(processes=process_num)
    res_getter = list()
    for i in range(process_num):
        res = pool.apply_async(func=func, args=args_list[i] if args_list else (),
                               kwds=kwargs_list[i] if kwargs_list else {})
        res_getter.append(res)
    pool.close()
    pool.join()
    results = list()
    for i in range(process_num):
        results.append(res_getter[i].get())
    return results


def dump_array(file, array, overwrite=True, sort_keys=False):
    if type(array) is not list:
        raise TypeError("Dict array not of valid type.")
    with open(file, 'w' if overwrite else 'a') as fp:
        for element in array:
            fp.write(json.dumps(element, sort_keys=sort_keys) + '\n')


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
