import argparse
import time
from collections import Counter
import numpy as np
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.pattern_utils as pu
import utils.date_utils as du
import utils.spacy_utils as su
import utils.array_utils as au
import utils.tweet_utils as tu
import math


def main(args):
    # p_range = [1, 2, 4, 8]
    # params = [(p1, p2, p3, p4, p5) for p1 in p_range for p2 in p_range
    #           for p3 in p_range for p4 in p_range for p5 in p_range]
    #
    # process_num = 20
    # param_num = len(params)
    # res_list = list()
    # for i in range(int(math.ceil(param_num / process_num))):
    #     res_list += fu.multi_process(make_comparision, params[i * process_num: (i + 1) * process_num])
    #     print('{:<4} / {} params processed'.format(min((i + 1) * process_num, param_num), param_num))
    # if not len(res_list) == len(params):
    #     raise ValueError('Error occurs')
    #
    # print(sorted(res_list, key=lambda x: x[0]))
    # return
    
    # import time
    # real_file = '/home/nfs/cdong/tw/src/clustering/data/events2016.txt'
    # block = fu.load_array(real_file)
    # twarr = fu.merge_list(block)[:5000]
    # print(len(twarr))
    # s = time.time()
    # tu.twarr_nlp(twarr)
    # print(time.time() - s)
    # return
    
    base = args.base
    files = args.files
    if not base or not fi.exists(base) or not fi.is_dir(base):
        base = fi.add_sep_if_needed(fi.pwd())
    if not files:
        files = fi.listchildren(base, children_type=fi.TYPE_FILE)
        
        real_file = '/home/nfs/cdong/tw/src/clustering/data/nonevents.txt'
        real_file = '/home/nfs/cdong/tw/src/clustering/data/events2016.txt'
        arr = fu.load_array(real_file)
        if type(arr[0]) is dict:
            block = au.array_partition(arr, [1] * 30)
        elif type(arr[0]) is list:
            block = arr
        else:
            block = None
        
        for twarr in block:
            twarr = tu.twarr_nlp(twarr)
            print(len(twarr), tu.twarr_similarity(twarr))
            
            # for tw in twarr:
            #     doc = tw.get(tk.key_spacy)
            #     pos = [t.tag_ for t in doc]
            #     pos_list.extend(pos)
            
            # for tw in twarr:
            #     ner_pos_tags = tw.get(tk.key_ner_pos)
            #     for text, e_type, pos in ner_pos_tags:
            #         if e_type not in {'GPE', 'ORG', 'LOC', 'FAC', 'GPE', }:
            #             continue
            #         text = text.lower()
            #         if text not in ner_dict:
            #             ner_dict[text] = {}
            #         if e_type not in ner_dict[text]:
            #             ner_dict[text][e_type] = 1
            #         else:
            #             ner_dict[text][e_type] += 1
            
            # print(sorted([item for item in ner_dict.items() if max(item[1].values()) > 5],
            #              key=lambda item: max(item[1].values()), reverse=True))
            # print(len(twarr))
            # print('\n\n')
            
            # block = fu.load_array(real_file)
            # for b_idx, twarr in enumerate(block):
            #     for idx, tw in enumerate(twarr):
            #         try:
            #             du.get_timestamp_form_created_at(tw[tk.key_created_at])
            #         except:
            #             print(real_file, b_idx, idx, tw[tk.key_created_at])


# def make_comparision(*prams):
#     event_blocks = fu.load_array('/home/nfs/cdong/tw/src/clustering/data/events2016.txt')
#     event_perform = []
#     for twarr in event_blocks:
#         twarr = tu.twarr_nlp(twarr)
#         event_perform.append(tu.twarr_similarity(twarr, *prams))
#
#     false_perform = []
#     false_twarr = fu.load_array('/home/nfs/cdong/tw/src/clustering/data/falseevents.txt')
#     block = au.array_partition(false_twarr, [1] * 30)
#     for twarr in block:
#         twarr = tu.twarr_nlp(twarr)
#         false_perform.append(tu.twarr_similarity(twarr, *prams))
#
#     return min(event_perform) - max(false_perform), min(event_perform), max(false_perform), \
#            np.mean(event_perform), np.mean(false_perform), prams


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('-b', action='store', dest='base', default='', help='list files to be parsed.')
    parser.add_argument('-f', action='append', dest='files', default=[], help='list files to be parsed.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
