import argparse
import time
from collections import Counter
import numpy as np

import utils.array_utils
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils
import utils.tweet_keys as tk
import utils.pattern_utils as pu
import utils.date_utils as du
import utils.spacy_utils as su
import utils.array_utils as au
import utils.tweet_utils as tu
import utils.timer_utils as tmu
import math


def main(args):
    """"""
    
    """ retweet chain """
    file ='/home/nfs/cdong/tw/seeding/Terrorist/queried/event2016.txt'
    twarr = utils.array_utils.merge_array(fu.load_array(file))
    rechain_dict = dict()
    for idx, tw in enumerate(twarr):
        twid, retwid = tw[tk.key_id], tu.in_reply_to(tw)
        if retwid is not None:
            if retwid not in rechain_dict:
                rechain_dict[retwid] = set()
            else:
                rechain_dict[retwid].add(retwid)
        else:
            if twid not in rechain_dict:
                rechain_dict[twid] = set()
            else:
                rechain_dict[twid].add(twid)
    print([len(chain) for chain in rechain_dict.values() if len(chain) >= 3])
    print(len(rechain_dict), len(twarr))
    return
    
    """  only to extract text """
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/'
    target = '/home/nfs/cdong/tw/seeding/Terrorist/queried/only_text/'
    subs = fi.listchildren(base, fi.TYPE_FILE)
    for sub in subs:
        twarr = fu.load_array(base + sub)
        textarr = [tw[tk.key_orgntext] for tw in twarr]
        fu.dump_array(target + sub, textarr)
    return
    
    """ test the vector similarity """
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
    
    """ see the similarity between clusters """
    # blocks = fu.load_array('/home/nfs/cdong/tw/seeding/Terrorist/queried/event2016.txt')
    # twarr = fu.merge_list(blocks)
    # label = fu.merge_list([[i for _ in range(len(blocks[i]))] for i in range(len(blocks))])
    #
    # label_distrb = Counter(label)
    # print('Topic num:{}, total tw:{}'.format(len(label_distrb), len(twarr)))
    # for idx, cluid in enumerate(sorted(label_distrb.keys())):
    #     print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
    #
    # from clustering.cluster_service import ClusterService
    # twarr = tu.twarr_nlp(twarr)
    # topic_vec_dict, sim_matrix = ClusterService.create_clusters_and_vectors(twarr, label)
    #
    # for idx, row in sim_matrix.iterrows():
    #     max_idx = np.argmax(row.values)
    #     max_col = row.index[max_idx]
    #     max_sim = row[max_col]
    #     print('maxsim ({}, {})={}'.format(idx, max_col, max_sim))
    # return
    
    """ pre-process the single data file """
    # real_file = '/home/nfs/cdong/tw/src/clustering/data/events2016.txt'
    # twarr = fu.merge_list(fu.load_array(real_file))
    # idx2rechain = [[] for _ in range(len(twarr))]
    # retwid2idx = dict([(tw[tk.key_id], idx) for idx, tw in enumerate(twarr)])
    # for idx, tw in enumerate(twarr):
    #     retwid = tu.in_reply_to(tw)
    #
    #     if retwid is None or retwid not in retwid2idx:
    #         continue
    #     idx2rechain[retwid2idx[retwid]].append(idx)
    # print(max([len(chain) for chain in idx2rechain]))
    # return
    
    """  """
    # base = args.base
    # files = args.files
    # if not base or not fi.exists(base) or not fi.is_dir(base):
    #     base = fi.add_sep_if_needed(fi.pwd())
    # if not files:
    #     files = fi.listchildren(base, children_type=fi.TYPE_FILE)
    #
    #     real_file = '/home/nfs/cdong/tw/src/clustering/data/nonevents.txt'
    #     real_file = '/home/nfs/cdong/tw/src/clustering/data/events2016.txt'
    #     arr = fu.load_array(real_file)
    #     if type(arr[0]) is dict:
    #         block = au.array_partition(arr, [1] * 30)
    #     elif type(arr[0]) is list:
    #         block = arr
    #     else:
    #         block = None
    #
    #     for twarr in block:
    #         twarr = tu.twarr_nlp(twarr)
    #         print(len(twarr), tu.twarr_vector_info(twarr, info_type='similarity'))
            
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


def test_spacy():
    import spacy
    tmu.check_time()
    nlp = spacy.load('en', vector=False)
    base = '/home/nfs/cdong/tw/origin/{}'
    sub_files = fi.listchildren(base.format(''), fi.TYPE_FILE)[:1]
    for file in sub_files:
        twarr = fu.load_array(base.format(file))
        textarr = [tw[tk.key_text] for tw in twarr]
        tmu.check_time(print_func=lambda dt: print('start nlp, len(textarr)={}'.format(len(textarr))))
        docarr = [doc for doc in nlp.pipe(textarr, n_threads=4)]
        tmu.check_time(print_func=lambda dt: print('nlp time elapsed {}'.format(dt)))
        for idx, doc in enumerate(docarr):
            print()
            for ent in doc.ents:
                print(ent.text, ent.label_)
            print()


if __name__ == '__main__':
    test_spacy()
