import os
import time
import json
import shutil
import multiprocessing as mp
import math


def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)


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


def merge_list(array):
    res = list()
    for item in array:
        res.extend(item)
    return res


def split_multi_format(array, process_num):
    block_size = math.ceil(len(array) / process_num)
    formatted_array = list()
    for i in range(process_num):
        formatted_array.append(array[i * block_size: (i + 1) * block_size])
    return formatted_array


def multi_process(func, args_list):
    """
    Do func in multiprocess way.
    :param func: To be executed within every
    :param args_list:
    :return:
    """
    process_num = len(args_list)
    pool = mp.Pool(processes=process_num)
    res_getter = list()
    for i in range(process_num):
        res = pool.apply_async(func=func, args=args_list[i])
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
