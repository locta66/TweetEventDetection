import math
from collections import Counter

import seeding.main2parser as main2parser
import utils.array_utils as au
from config.configure import getcfg
import utils.function_utils as fu
import utils.tweet_utils as tu
import utils.tweet_keys as tk
import preprocess.tweet_filter as tflt


# def exec_temp(parser):
#     """find tws that classifier fail to recognize on dzs_neg_data, and split them into blocks"""
#     event_extractor = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
#     dzs_neg_data = '/home/nfs/cdong/tw/testdata/non_pos_tweets.sum'
#     dzs_neg_twarr = fu.load_array(dzs_neg_data)
#     preds = event_extractor.make_classification(dzs_neg_twarr)
#     false_pos_twarr = [dzs_neg_twarr[idx] for idx, pred in enumerate([pred for pred in preds]) if pred > 0.5]
#     # false_pos_twarr_blocks = au.array_partition(false_pos_twarr, [1] * 10)
#     fu.dump_array('falseevents.txt', false_pos_twarr)
#
#     """query for pos events into blocks"""
#     data_path = getcfg().summary_path
#     log_path = '/home/nfs/cdong/tw/testdata/yli/queried_events_with_keyword/'
#     twarr_blocks = main2parser.query_per_query_multi(data_path, parser.seed_query_list)
#     print('query done, {} events, {} tws'.format(len(twarr_blocks), sum([len(arr) for arr in twarr_blocks])))
#     event_id2info = dict()
#     tu.start_ner_service(pool_size=16, classify=True, pos=True)
#     for i in range(len(parser.seed_query_list)):
#         event_id2info[i] = dict()
#         twarr_blocks[i] = tu.twarr_ner(twarr_blocks[i])
#         query_i = parser.seed_query_list[i]
#         file_name_i = query_i.all[0].strip('\W') + '_' + '-'.join(query_i.since) + '.sum'
#         event_id2info[i]['filename'] = file_name_i
#         event_id2info[i]['all'] = [w.strip('\W') for w in query_i.all]
#         event_id2info[i]['any'] = [w.strip('\W') for w in query_i.any]
#         fu.dump_array(log_path + file_name_i, twarr_blocks[i])
#     fu.dump_array(log_path + 'event_id2info.txt', [event_id2info])
#     tu.end_ner_service()
#     print(event_id2info)
#     print('ner done')
#     fu.dump_array('events.txt', twarr_blocks)
#
#     """splitting dzs_neg_data into blocks"""
#     twarr_blocks = list()
#     dzs_neg_data = '/home/nfs/cdong/tw/testdata/non_pos_tweets.sum'
#     dzs_neg_parts = au.array_partition(fu.load_array(dzs_neg_data), (1, 1, 1, 1))
#     for dzs_part in dzs_neg_parts:
#         twarr_blocks.append(au.random_array_items(dzs_part, 300))
#     my_neg_data = '/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_counter.sum'
#     my_neg_parts = au.array_partition(fu.load_array(my_neg_data), (1, 1, 1, 1))
#     for my_part in my_neg_parts:
#         twarr_blocks.append(au.random_array_items(my_part, 300))
#
#     print('tweet distribution ', [len(twarr) for twarr in twarr_blocks], '\n\rtotal cluster', len(twarr_blocks))
#     fu.dump_array('nonevents.txt', twarr_blocks)


def lbarr_of_twarr(twarr):
    return [tw[tk.key_event_label] for tw in twarr]


def order_twarr_through_time():
    print('data source : normal')
    event_blocks = fu.load_array('./data/events2016.txt')
    false_event_twarr = fu.load_array('./data/false_pos_events.txt')
    event_blocks.append(false_event_twarr)
    for block_idx, block in enumerate(event_blocks):
        for tw in block:
            tw[tk.key_event_label] = block_idx
    twarr = au.merge_array(event_blocks)
    tflt.filter_twarr_dup_id(twarr)
    
    def random_idx_for_item(item_arr, dest_item):
        from numpy import random
        def sample(prob):
            return random.rand() < prob
        
        non_dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] not in dest_item]
        dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] in dest_item]
        non_dest_cnt = dest_cnt = 0
        res = list()
        while len(non_dest_item_idx) > non_dest_cnt and len(dest_item_idx) > dest_cnt:
            if sample((len(dest_item_idx) - dest_cnt) /
                      (len(dest_item_idx) - dest_cnt + len(non_dest_item_idx) - non_dest_cnt)):
                res.append(dest_item_idx[dest_cnt])
                dest_cnt += 1
            else:
                res.append(non_dest_item_idx[non_dest_cnt])
                non_dest_cnt += 1
        while len(non_dest_item_idx) > non_dest_cnt:
            res.append(non_dest_item_idx[non_dest_cnt])
            non_dest_cnt += 1
        while len(dest_item_idx) > dest_cnt:
            res.append(dest_item_idx[dest_cnt])
            dest_cnt += 1
        return res
    
    idx_time_order = tu.rearrange_idx_by_time(twarr)
    twarr = [twarr[idx] for idx in idx_time_order]
    lbarr = lbarr_of_twarr(twarr)
    idx_random_item = random_idx_for_item(lbarr, {max(lbarr)})
    twarr = [twarr[idx] for idx in idx_random_item]
    return twarr


def split_into_batches(twarr, batch_size):
    """ cutting twarr and label into batches of batch_size """
    max_len = len(twarr)
    batch_num = int(math.ceil(len(twarr) / batch_size))
    tw_batches = [[twarr[j] for j in range(i*batch_size, min((i+1)*batch_size, max_len))] for i in range(batch_num)]
    twarr_info(au.merge_array(tw_batches))
    return tw_batches


def make_tw_batches(batch_size):
    attr_set = {tk.key_text, tk.key_event_label, tk.key_id, tk.key_created_at}
    ordered_twarr = order_twarr_through_time()
    for tw in ordered_twarr:
        for k in list(tw.keys()):
            if k not in attr_set:
                tw.pop(k)
    tw_batches = split_into_batches(ordered_twarr, batch_size)
    fu.dump_array('./data/batches.txt', tw_batches)


def get_tw_batches():
    tw_batches = fu.load_array('./data/batches.txt')
    twarr_info(au.merge_array(tw_batches))
    return tw_batches


def twarr_info(twarr):
    lbarr = lbarr_of_twarr(twarr)
    label_distrb = Counter(lbarr)
    for idx, cluid in enumerate(sorted(label_distrb.keys())):
        print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
    print('\nTopic num: {}, total tw: {}'.format(len(label_distrb), len(twarr)))


# def create_batches_through_time(batch_size):
#     print('data source : normal')
#     event_blocks = fu.load_array('./data/events2016.txt')
#     # false_event_twarr = fu.load_array('./data/falseevents.txt')[:200]
#     # event_blocks = fu.load_array('./data/events.txt')
#     false_event_twarr = fu.load_array('./data/falseevents.txt')
#     event_blocks.append(false_event_twarr)
#
#     for block_idx, block in enumerate(event_blocks):
#         for tw in block:
#             tw.setdefault(tk.key_event_label, block_idx)
#
#     twarr = au.merge_array(event_blocks)
#     lbarr = au.merge_array([[i for _ in range(len(event_blocks[i]))] for i in range(len(event_blocks))])
#
#     dup_idx_list = tflt.twarr_dup_id(twarr)
#     for idx in range(len(dup_idx_list) - 1, -1, -1):
#         twarr.pop(idx)
#         lbarr.pop(idx)
#     # twarr = [twarr[idx] for idx in range(len(twarr)) if idx not in dup_idx_list]
#     # lbarr = [lbarr[idx] for idx in range(len(lbarr)) if idx not in dup_idx_list]
#
#     label_distrb = Counter(lbarr)
#     for idx, cluid in enumerate(sorted(label_distrb.keys())):
#         print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
#     print('\nTopic num: {}, total tw: {}'.format(len(label_distrb), len(twarr)))
#
#     idx_time_order = tu.rearrange_idx_by_time(twarr)
#     twarr = [twarr[idx] for idx in idx_time_order]
#     lbarr = [lbarr[idx] for idx in idx_time_order]
#
#     # for idx in range(len(twarr) - 1):
#     #     if du.get_timestamp_form_created_at(twarr[idx][TweetKeys.key_created_at].strip()) > \
#     #             du.get_timestamp_form_created_at(twarr[idx + 1][TweetKeys.key_created_at].strip()):
#     #         raise ValueError('wrong')
#
#     def random_idx_for_item(item_arr, dest_item):
#         from numpy import random
#         def sample(prob):
#             return random.rand() < prob
#         non_dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] not in dest_item]
#         dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] in dest_item]
#         non_dest_cnt = dest_cnt = 0
#         res = list()
#         while len(non_dest_item_idx) > non_dest_cnt and len(dest_item_idx) > dest_cnt:
#             if sample((len(dest_item_idx) - dest_cnt) /
#                       (len(dest_item_idx) - dest_cnt + len(non_dest_item_idx) - non_dest_cnt)):
#                 res.append(dest_item_idx[dest_cnt])
#                 dest_cnt += 1
#             else:
#                 res.append(non_dest_item_idx[non_dest_cnt])
#                 non_dest_cnt += 1
#         while len(non_dest_item_idx) > non_dest_cnt:
#             res.append(non_dest_item_idx[non_dest_cnt])
#             non_dest_cnt += 1
#         while len(dest_item_idx) > dest_cnt:
#             res.append(dest_item_idx[dest_cnt])
#             dest_cnt += 1
#         return res
#
#     idx_rearrange = random_idx_for_item(lbarr, {len(event_blocks) - 1})
#     twarr = [twarr[idx] for idx in idx_rearrange]
#     lbarr = [lbarr[idx] for idx in idx_rearrange]
#     """ cutting twarr and label into batches of batch_size """
#     # idx_parts = au.index_partition(twarr, [1] * int(len(twarr) / batch_size), random=False)
#     # tw_batches = [[twarr[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
#     # lb_batches = [[label[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
#
#     full_idx = [i for i in range(len(twarr))]
#     batch_num = int(math.ceil(len(twarr) / batch_size))
#     tw_batches = [[twarr[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
#     lb_batches = [[lbarr[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
#     # label = fu.merge_list(lb_batches)
#     # print(len(twarr), len(label))
#     # label_distrb = Counter(label)
#     # print('\n\nTopic num:{}, total tw:{}'.format(len(label_distrb), len(twarr)))
#     # for idx, cluid in enumerate(sorted(label_distrb.keys())):
#     #     print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
#     # print()
#     lbarr = au.merge_array(lb_batches)
#     label_distrb = Counter(lbarr)
#     for idx, cluid in enumerate(sorted(label_distrb.keys())):
#         print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
#     print('\nTopic num: {}, total tw: {}'.format(len(label_distrb), len(twarr)))
#     exit()
#
#     return tw_batches, lb_batches


# def create_korea_batches_through_time(batch_size):
#     print('data source : korea')
#     false_twarr = fu.load_array('./data/falseevents.txt')
#     event_blocks = fu.load_array('./data/events.txt')
#     event_blocks.append(false_twarr)
#     non_korea_twarr = au.merge_array(event_blocks)
#     non_korea_twarr = sorted(non_korea_twarr, key=lambda item: item.get(tk.key_id))
#     twarr_blocks = fu.load_array('/home/nfs/cdong/tw/seeding/NorthKorea/korea.json')
#     twarr_blocks.append(non_korea_twarr)
#     for idx, twarr in enumerate(twarr_blocks):
#         print(idx, len(twarr), end=' -> ')
#         tflt.filter_twarr_dup_id(twarr)
#         print(len(twarr))
#     """ allocate a label for every tweet """
#     for idx, twarr in enumerate(twarr_blocks):
#         for tw in twarr:
#             tw.setdefault(tk.key_event_label, idx)
#     twarr = au.merge_array(twarr_blocks)
#     """ rearrange indexes """
#     def random_idx_for_item(item_arr, dest_item):
#         from numpy import random
#         def sample(prob):
#             return random.rand() < prob
#         non_dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] not in dest_item]
#         dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] in dest_item]
#         non_dest_cnt = dest_cnt = 0
#         res = list()
#         while len(non_dest_item_idx) > non_dest_cnt and len(dest_item_idx) > dest_cnt:
#             if sample((len(dest_item_idx) - dest_cnt) /
#                       (len(dest_item_idx) - dest_cnt + len(non_dest_item_idx) - non_dest_cnt)):
#                 res.append(dest_item_idx[dest_cnt])
#                 dest_cnt += 1
#             else:
#                 res.append(non_dest_item_idx[non_dest_cnt])
#                 non_dest_cnt += 1
#         while len(non_dest_item_idx) > non_dest_cnt:
#             res.append(non_dest_item_idx[non_dest_cnt])
#             non_dest_cnt += 1
#         while len(dest_item_idx) > dest_cnt:
#             res.append(dest_item_idx[dest_cnt])
#             dest_cnt += 1
#         return res
#     lbarr = [tw.get(tk.key_event_label) for tw in twarr]
#     idx_rearrange = random_idx_for_item(lbarr, {max(lbarr)})
#     twarr = [twarr[idx] for idx in idx_rearrange]
#     """ full split twarr & label """
#     full_idx = [i for i in range(len(twarr))]
#     batch_num = int(math.ceil(len(twarr) / batch_size))
#     tw_batches = [[twarr[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
#     lb_batches = [[tw.get(tk.key_event_label) for tw in tw_batch] for tw_batch in tw_batches]
#     return tw_batches, lb_batches


if __name__ == '__main__':
    make_tw_batches(100)
