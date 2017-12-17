import os

from Pattern import get_pattern
from JsonParser import JsonParser
import FunctionUtils as fu

unzip_cmd = pos_cmd = ner_cmd = ''


def append_slash_if_necessary(path):
    if not path.endswith(os.path.sep):
        path += os.path.sep
    return path


def make_dirs_if_not_exists(dir_name):
    if not type(dir_name) is str:
        raise ValueError('Not valid directory description token')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def base_name(abspath):
    return os.path.basename(abspath)


def get_parent_path(path):
    return os.path.dirname(path)


def is_dir(path):
    return os.path.isdir(path)


def is_file(file):
    return os.path.isfile(file)


def remove_files(files):
    if type(files) is list:
        for f in files:
            remove_file(f)
    else:
        raise TypeError("File descriptor array not an expected type")


def remove_file(file):
    if type(file) is str:
        if os.path.exists(file):
            # print('removing file', file)
            os.remove(file)
    else:
        raise TypeError("File descriptor not an expected type")


def listchildren(directory, children_type='dir'):
    if children_type not in ['dir', 'file', 'all']:
        print('listchildren() : Incorrect children type')
        return list()
    directory = append_slash_if_necessary(directory)
    children = sorted(os.listdir(directory))
    if children_type == 'all':
        return children
    res = list()
    for child in children:
        child_path = directory + child
        if not os.path.exists(child_path):
            print('listchildren() : Invalid path')
            continue
        is_dir = os.path.isdir(child_path)
        if children_type == 'dir' and is_dir:
            res.append(child)
        elif children_type == 'file' and not is_dir:
            res.append(child)
    return res


def iterate_file_tree(root_path, func, *args, **kwargs):
    """
    Iterate file tree under the root path,, and invoke func in those directories with no sub-directories.
    :param root_path: Current location of tree iteration.
    :param func: A callback function.
    :param args: Just to pass over outer params for func.
    :param kwargs: Just to pass over outer params for func.
    """
    subdirs = listchildren(root_path, children_type='dir')
    if not subdirs:
        func(root_path, *args, **kwargs)
    else:
        for subdir in subdirs:
            iterate_file_tree(root_path + subdir + os.path.sep, func, *args, **kwargs)


# def unzip_files_in_path(path, *args, **kwargs):
#     path = append_slash_if_necessary(path)
#     subfiles = listchildren(path, children_type='file')
#     for subfile in subfiles:
#         unzip_file(path + subfile)
#
#
# def unzip_file(file_path):
#     if not judgetype(".\.bz2$", file_path):
#         return
#     command = unzip_cmd + file_path
#     try:
#         p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                              stderr=subprocess.PIPE, close_fds=True)
#         stdout, stderr = p.communicate('')
#         print(file_path + ' unzip done')
#     except:
#         print('unzip file error')


def summary_files_in_path(file_path, *args, **kwargs):
    """
    To read all .json files under the json_path, and extract tweets of interest from them.
    :param file_path: A directory with no sub-directories.
    :param args: Just to pass over outer params for func.
    :param kwargs: Just to pass over outer params for func.
    :return:
    """
    # granularity: [-13:]--hour [-13:-3]--day [-13:-5]--month
    # ymdh refers to the short of "year-month-date-hour"
    json_ymdh_str = get_pattern().get_parent_path(file_path)[-13:]
    json_ymdh_arr = get_pattern().full_split_nondigit(json_ymdh_str)
    if not is_target_ymdh(json_ymdh_arr):
        return
    
    file_path = append_slash_if_necessary(file_path)
    summary_path = append_slash_if_necessary(kwargs['summary_path'])
    summary_name = '_'.join(json_ymdh_arr)
    summary_file = summary_path + summary_name + '.sum'
    remove_ymdh_from_path(summary_path, summary_name)
    subfiles = listchildren(file_path, children_type='file')
    
    file_list = fu.split_multi_format([(file_path + subfile) for subfile in subfiles], process_num=15)
    twarr_blocks = fu.multi_process(summary_zipped_tweets_multi,
                                 [(file_list_slice,) for file_list_slice in file_list])
    twarr = fu.merge_list(twarr_blocks)
    
    if twarr:
        fu.dump_array(summary_file, twarr)
    print(summary_file, 'written')


def summary_zipped_tweets_multi(file_list):
    j_parser = JsonParser()
    twarr = list()
    for file in file_list:
        if not file.endswith(".bz2"):
            continue
        twarr.extend(j_parser.read_tweet_from_bz2_file(file))
    return twarr


def summary_unzipped_tweets_multi(file_list):
    j_parser = JsonParser()
    twarr = list()
    for file in file_list:
        if not file.endswith(".json"):
            continue
        twarr.extend(j_parser.read_tweet_from_json_file(file))
    return twarr


def remove_ymdh_from_path(summary_path, ymdh_file_name):
    summary_path = append_slash_if_necessary(summary_path)
    subfiles = listchildren(summary_path, children_type='file')
    for subfile in subfiles:
        if ymdh_file_name in subfile:
            remove_file(summary_path + subfile)


# def simplify_files_multi(from_dir, to_dir):
#     import FunctionUtils as fu
#     import ArrayUtils as au
#     file_list = listchildren(from_dir, children_type='file')[:6]
#     file_lists = au.array_partition(file_list, [1] * 2)
#     fu.multi_process(simplify_files, [(from_dir, f, to_dir) for f in file_lists])


# def simplify_files(from_dir, file_list, to_dir):
#     import FunctionUtils as fu
#     tweet_desired_attrs = ['created_at', 'timestamp_ms', 'id', 'text', 'place', 'user',
#                             'retweet_count', 'favorite_count', 'in_reply_to_status_id', 'in_reply_to_user_id', ]
#     user_desired_attrs = ['id', 'created_at', 'time_zone', 'location', 'favourites_count',
#                            'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'verified', ]
#     def attribute_filter(target_dict, attr_list):
#         for attr in list(target_dict.keys()):
#             if attr not in attr_list:
#                 target_dict.pop(attr)
#         return target_dict
#
#     for file in file_list:
#         print(from_dir + file)
#         twarr = fu.load_array(from_dir + file)
#         for tw in twarr:
#             attribute_filter(tw, tweet_desired_attrs)
#             attribute_filter(tw['user'], user_desired_attrs)


# def pos_of_summary(summary_file):
#     name_without_postfix = summary_file[0:-4]
#     pos_file = name_without_postfix + '.pos'
#     command = '%s %s > %s' % (pos_cmd, summary_file, pos_file)
#     try:
#         subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE, close_fds=True).communicate('')
#         print(summary_file + ' pos done')
#     except:
#         print('pos error')
#         return ''
#     return pos_file
#
#
# def tokens_of_pos(pos_file):
#     name_without_postfix = pos_file[0:-4]
#     pos_token_file = name_without_postfix + '.tkn'
#     if os.path.exists(pos_token_file):
#         remove_file(pos_token_file)
#     with open(pos_file, 'r') as posfp, open(pos_token_file, 'a') as tknfp:
#         for line in posfp.readlines():
#             tokenized = line.split('\t', 4)[0]
#             tknfp.write(tokenized + '\n')
#     return pos_token_file
#
#
# def ner_of_tokens(token_file):
#     name_without_postfix = token_file[0:-4]
#     ner_file = name_without_postfix + '.ner'
#     command = '%s %s -o %s' % (ner_cmd, token_file, ner_file)
#     try:
#         subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE, close_fds=True).communicate('')
#         print(token_file + ' ner done')
#     except:
#         print('ner error')
#         return ''
#     return ner_file
#
#
# def combine_sum_pos_ner(summary_file, pos_file, ner_file):
#     name_without_postfix = summary_file[0:-4]
#     combine_file = name_without_postfix + '.txt'
#     with open(summary_file) as sumfp, open(pos_file) as posfp, open(ner_file) as nerfp:
#         sum = sumfp.readlines()
#         pos = posfp.readlines()
#         ner = nerfp.readlines()
#         if not (len(sum) == len(pos) and len(sum) == len(ner)):
#             print('sum-pos-ner files line count inconsistent')
#             return
#         p = get_pattern()
#         json_parser = JsonParser()
#         tw_arr = []
#         for i in range(len(sum)):
#             tw = json_parser.parse_text(p.remove_endline(sum[i]), filtering=False)
#             tokens, pos_labels = p.remove_endline(pos[i]).split('\t', 4)[0:2]
#             tw['text'] = tokens
#             tw['pos'] = pos_labels
#             tw['ner'] = p.remove_endline(ner[i])
#             # if not len(tw['ner'].split(' ')) == len(tw['pos'].split(' ')):
#             #     print('ner num:', len(tw['ner'].split(' ')), 'pos num:', len(tw['pos'].split(' ')))
#             #     print(tw['pos'])
#             #     print(tw['ner'])
#             #     print('\n')
#             tw_arr.extend([tw])
#         json_parser.dump_json_arr_into_file(tw_arr, combine_file, mode='renew')
#         return combine_file
#
#
# def validate_line_consistency(file1, file2):
#     def line_of_file(file):
#         with open(file, 'r') as fp:
#             return len(fp.readlines())
#     cnt1 = line_of_file(file1)
#     cnt2 = line_of_file(file2)
#     if not cnt1 == cnt2:
#         print(file1, ':', cnt1, 'lines')
#         print(file2, ':', cnt2, 'lines')
#         return False
#     else:
#         return True


def is_target_ymdh(ymdh_arr):
    # ymdh_arr resembles ['201X', '0X', '2X', '1X']
    year = int(ymdh_arr[0])
    month = int(ymdh_arr[1])
    date = int(ymdh_arr[2])
    hour = int(ymdh_arr[3])
    return True
    # return 4 == month and date <= 2
    # import datetime
    # tw_time = datetime.datetime.strptime('-'.join(ymdh_arr[0:3]), '%Y-%m-%d')
    # start_time = datetime.datetime.strptime('2016-06-08', '%Y-%m-%d')
    # return (tw_time - start_time).days >= 0
