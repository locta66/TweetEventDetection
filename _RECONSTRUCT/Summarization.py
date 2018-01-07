import re
import json
import bz2file

from IdFreqDict import IdFreqDict
import TweetFilter as tflt
import TweetKeys as tk
import ArrayUtils as au
import FunctionUtils as fu
import FileIterator as fi
import PatternUtils as pu
import ArkServiceProxy as ark


def summary_files_in_path(from_path, into_path=None):
    """ Read all .json under file_path, extract tweets from them into a file under summary_path. """
    # [-13:]--hour [-13:-3]--day [-13:-5]--month,ymdh refers to the short of "year-month-date-hour"
    from_path = fi.add_sep_if_needed(from_path)
    file_ymdh_arr = pu.split_digit_arr(fi.get_parent_path(from_path)[-13:])
    if not is_target_ymdh(file_ymdh_arr):
        return
    
    into_file = '{file_path}{file_name}'.format(
        file_path=fi.add_sep_if_needed(into_path), file_name='_'.join(file_ymdh_arr) + '.sum')
    fi.remove_file(into_file)
    subfiles = fi.listchildren(from_path, children_type=fi.TYPE_FILE)
    file_block = fu.split_multi_format([(from_path + subfile) for subfile in subfiles], process_num=20)
    twarr_blocks = fu.multi_process(sum_files, [(file_list, tflt.FILTER_LEVEL_LOW) for file_list in file_block])
    twarr = fu.merge_list(twarr_blocks)
    if twarr:
        fu.dump_array(into_file, twarr, overwrite=True)


def sum_files(file_list, filter_level):
    res_twarr = list()
    for file in file_list:
        twarr = load_twarr_from_bz2(file) if file.endswith('.bz2') else \
                fu.load_array(file) if file.endswith('.json') else None
        twarr = tflt.filter_twarr(twarr, filter_level)
        res_twarr.extend(twarr)
    return res_twarr


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


def is_target_ymdh(ymdh_arr):
    # ymdh_arr resembles ['201X', '0X', '2X', '1X']
    year = int(ymdh_arr[0])
    month = int(ymdh_arr[1])
    date = int(ymdh_arr[2])
    hour = int(ymdh_arr[3])
    # return 4 == month and date <= 2
    # import datetime
    # tw_time = datetime.datetime.strptime('-'.join(ymdh_arr[0:3]), '%Y-%m-%d')
    # start_time = datetime.datetime.strptime('2016-06-08', '%Y-%m-%d')
    # return (tw_time - start_time).days >= 0
    return True


def get_tokens_multi(file_path):
    file_path = fi.add_sep_if_needed(file_path)
    subfiles = au.random_array_items(fi.listchildren(file_path, children_type=fi.TYPE_FILE), 400)
    file_list_block = fu.split_multi_format([(file_path + subfile) for subfile in subfiles], process_num=20)
    res_list = fu.multi_process(get_tokens, [(file_list, ) for file_list in file_list_block])
    id_freq_dict, total_doc_num = IdFreqDict(), 0
    for ifd, doc_num in res_list:
        total_doc_num += doc_num
        id_freq_dict.merge_freq_from(ifd)
    print('total_doc_num', total_doc_num, 'total vocabulary_size', id_freq_dict.vocabulary_size())
    id_freq_dict.drop_words_by_condition(2)
    id_freq_dict.dump_dict('temp.csv')


def get_tokens(file_list):
    id_freq_dict, total_doc_num = IdFreqDict(), 0
    for file in file_list:
        twarr = fu.load_array(file)
        total_doc_num += len(twarr)
        for tw in twarr:
            tokens = re.findall(r'[a-zA-Z_#\-]{3,}', tw[tk.key_text].lower())
            real_tokens = list()
            for token in tokens:
                real_tokens.extend(pu.segment(token)) if len(token) >= 16 else [token]
            for token in real_tokens:
                if not pu.is_stop_word(token) and pu.has_azAZ(token) and 3 <= len(token) <= 16:
                    id_freq_dict.count_word(token)
    return id_freq_dict, total_doc_num


def get_semantic_tokens_multi(file_path):
    pos_type_info = {
        ark.is_proper_noun: {'ifd': IdFreqDict(), 'file': 'prop_noun'},
        ark.is_common_noun: {'ifd': IdFreqDict(), 'file': 'common_noun'},
        ark.is_verb: {'ifd': IdFreqDict(), 'file': 'verb'},
        ark.is_hashtag: {'ifd': IdFreqDict(), 'file': 'hashtag'},
    }
    total_doc_num = 0
    file_path = fi.add_sep_if_needed(file_path)
    subfiles = au.random_array_items(fi.listchildren(file_path, children_type=fi.TYPE_FILE), 400)
    file_list_block = fu.split_multi_format([(file_path + subfile) for subfile in subfiles], process_num=20)
    res_list = fu.multi_process(get_semantic_tokens, [(file_list, ) for file_list in file_list_block])
    for func2ifd, doc_num in res_list:
        total_doc_num += doc_num
        for func in func2ifd.keys():
            pos_type_info[func]['ifd'].merge_freq_from(func2ifd[func]['ifd'])
    print('total_doc_num', total_doc_num)
    for func in pos_type_info.keys():
        ifd, file_name = pos_type_info[func]['ifd'],  pos_type_info[func]['file']
        ifd.drop_words_by_condition(3)
        ifd.dump_dict(file_name + '.csv')
        print(file_name, ifd.vocabulary_size())


def get_semantic_tokens(file_list):
    pos_type_info = {
        ark.is_proper_noun: {'ifd': IdFreqDict()},
        ark.is_common_noun: {'ifd': IdFreqDict()},
        ark.is_verb: {'ifd': IdFreqDict()},
        ark.is_hashtag: {'ifd': IdFreqDict()},
    }
    total_doc_num = 0
    for file in file_list:
        twarr = ark.twarr_ark(fu.load_array(file))
        total_doc_num += len(twarr)
        pos_tokens = list()
        for tw in twarr:
            pos_tokens += tw[tk.key_ark]
        for pos_token in pos_tokens:
            print(pos_token)
            for func in pos_type_info.keys():
                word = pos_token[0]
                if len(word) <= 2 or pu.is_stop_word(word) or not pu.has_azAZ(word):
                    continue
                if func(pos_token):
                    pos_type_info[func]['ifd'].count_word(pos_token[0].lower())
                    continue
    return pos_type_info, total_doc_num


# sep = os.path.sep
# def load_twarr_from_bz2_list(bz2_file_list):
#     print(len(bz2_file_list))
#     twarr = list()
#     for bz2_file in bz2_file_list:
#         if not bz2_file.endswith(".bz2"):
#             continue
#         twarr.extend(load_twarr_from_bz2(bz2_file))
#     return twarr
# def load_twarr_from_bz2_multi(bz2_list, p_num=15):
#     list_per_p = fu.split_multi_format(bz2_list, min(p_num, len(bz2_list)))
#     return fu.multi_process(load_twarr_from_bz2_list, [(l, ) for l in list_per_p])
# def tw_file_date(file_or_dir):
#     if fi.is_dir(file_or_dir):
#         dir_name = file_or_dir[:file_or_dir.rfind('\\')] if file_or_dir.endswith('\\') else file_or_dir
#         timestr = dir_name[-15:]
#     elif fi.is_file(file_or_dir):
#         file_name = file_or_dir
#         timestr = fi.base_name(file_name)[:file_name.rfind('.')]
#     else:
#         raise ValueError('file_or_dir wrong format:', file_or_dir)
#     return timestr
# def list_raw_data(data_path, start_ymdh, end_ymdh):
#     """ used for the purpose of special file structure, which is organized as the tree
#         data_path/year_month/date/hour/minute.bz2
#         :return A list of sublist, where each sublist contains the data within one day """
#     ymd = [ym + sep + d + sep for ym in fi.listchildren(data_path, children_type='dir')
#            for d in fi.listchildren(data_path + ym + sep, children_type='dir')
#            if du.is_target_ymdh(pu.split_digit_arr(ym + sep + d), start_ymdh, end_ymdh)]
#     ymd_hM = [[data_path + ymd_ + h + sep + M for h in fi.listchildren(data_path + ymd_, children_type='dir')
#                for M in fi.listchildren(data_path + ymd_ + sep + h, children_type='file')] for ymd_ in ymd]
#     for idx in range(len(ymd_hM) - 1, -1, -1):
#         if not ymd_hM[idx]:
#             ymd_hM.pop(idx)
#     return ymd_hM
# # data_path = getconfig().data_path
# # start_ymdh, end_ymdh = ['2016', '03', '2'], ['2016', '03', '6']
# # bz2_lists = list_raw_data(data_path, start_ymdh, end_ymdh)
# # for bz2_list_per_day in bz2_lists:
# #     load_twarr_from_bz2_multi(bz2_list_per_day[:150], p_num=15)
