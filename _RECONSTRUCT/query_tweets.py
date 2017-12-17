import __init__

import json
import bz2file

from Configure import getconfig
import ArrayUtils as au
import DateUtils as du
import FileIterator as fi
import FunctionUtils as fu
import TweetKeys as tk
import os
import re

sep = os.path.sep


def load_twarr_from_bz2(bz2_file):
    fp = bz2file.open(bz2_file, 'r')
    twarr = list()
    for line in fp.readlines():
        line = line.decode('utf8')
        try:
            json_obj = json.loads(line)
            twarr.append(json_obj)
        except:
            print('error when parsing tweet')
            continue
    fp.close()
    return twarr


def load_twarr_from_bz2_list(bz2_file_list):
    print(len(bz2_file_list))
    twarr = list()
    for bz2_file in bz2_file_list:
        if not bz2_file.endswith(".bz2"):
            continue
        twarr.extend(load_twarr_from_bz2(bz2_file))
    return twarr


def load_twarr_from_bz2_multi(bz2_list, p_num=15):
    print(bz2_list)
    list_per_p = fu.split_multi_format(bz2_list, min(p_num, len(bz2_list)))
    fu.multi_process(load_twarr_from_bz2_list, [(l, ) for l in list_per_p])


def split_digit_arr(string):
    return [str_ for str_ in re.split('[^\d]', string) if re.findall('^\d+$', str_)]


def tw_file_date(file_or_dir):
    if fi.is_dir(file_or_dir):
        dir_name = file_or_dir[:file_or_dir.rfind('\\')] if file_or_dir.endswith('\\') else file_or_dir
        timestr = dir_name[-15:]
    elif fi.is_file(file_or_dir):
        file_name = file_or_dir
        timestr = fi.base_name(file_name)[:file_name.rfind('.')]
    else:
        raise ValueError('file_or_dir wrong format:', file_or_dir)
    return timestr


def list_raw_data(data_path, start_ymdh, end_ymdh):
    """ used for the purpose of special file structure, which is organized as the tree
        data_path/year_month/date/hour/minute.bz2
        :return A list of sublist, where each sublist contains the data within one day """
    ymd = [ym + sep + d + sep for ym in fi.listchildren(data_path, children_type='dir')
           for d in fi.listchildren(data_path + ym + sep, children_type='dir')
           if du.is_target_ymdh(split_digit_arr(ym + sep + d), start_ymdh, end_ymdh)]
    ymd_hM = [[data_path + ymd_ + h + sep + M for h in fi.listchildren(data_path + ymd_, children_type='dir')
               for M in fi.listchildren(data_path + ymd_ + sep + h, children_type='file')] for ymd_ in ymd]
    for idx in range(len(ymd_hM) - 1, -1, -1):
        if not ymd_hM[idx]:
            ymd_hM.pop(idx)
    return ymd_hM


data_path = getconfig().data_path
start_ymdh, end_ymdh = ['2016', '03', '2'], ['2016', '03', '6']
bz2_lists = list_raw_data(data_path, start_ymdh, end_ymdh)
for bz2_list_per_day in bz2_lists:
    load_twarr_from_bz2_multi(bz2_list_per_day[:150], p_num=15)


