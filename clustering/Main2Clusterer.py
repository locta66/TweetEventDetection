from collections import Counter

import numpy as np
import Main2Parser
import TweetKeys
import FunctionUtils as fu
import ArrayUtils as au
import TweetUtils as tu
import DateUtils as du
from ArrayUtils import roc_auc
from Configure import getconfig
from EventExtractor import EventExtractor
from EventTrainer import EventTrainer
from FunctionUtils import sync_real_time_counter


@sync_real_time_counter('query')
def exec_query(data_path, parser):
    Main2Parser.exec_query(data_path, parser)


def exec_ner(parser):
    Main2Parser.exec_ner(parser)


def exec_classification(seed_parser, test_parser):
    event_extractor = EventExtractor(seed_parser.get_dict_file_name(), seed_parser.get_param_file_name())
    
    # twarr = fI.load_array(test_parser.get_query_result_file_name())
    # pos_pred = event_extractor.make_classification(twarr)
    # print(recall([(pred[0], 1) for pred in pos_pred], [i / 10 for i in range(1, 10)]))
    
    pos_twarr = fu.load_array(getconfig().pos_data_file)
    non_pos_twarr = fu.load_array(getconfig().non_pos_data_file)
    pos_pred = event_extractor.make_classification(pos_twarr)
    non_pos_pred = event_extractor.make_classification(non_pos_twarr)
    scores = [pred[0] for pred in pos_pred] + [pred[0] for pred in non_pos_pred]
    labels = [1 for _ in pos_pred] + [0 for _ in non_pos_pred]
    print(roc_auc(labels, scores))
    # for idx, pred in enumerate(pos_pred):
    #     if pred < 0.2:
    #         print(pos_twarr[idx][TweetKeys.key_origintext], '\n---\n')


def exec_cluster(parser):
    # twarr, label = load_clusters_and_labels()
    # print('Label distribution ', list(Counter(label).values()), 'total cluster', Counter(label).__len__(),
    #       '\ntotal tweet num ', len(twarr))
    
    ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    # tw_topic_arr, cluster_pred = ee.LECM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDMM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDPMM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDMM_twarr_hashtag_with_label(twarr, label)
    # ee.semantic_cluster_with_label(twarr, label)
    
    tw_batches, lb_batches = create_batches_through_time()
    print('Label distribution ', list(Counter(fu.merge_list(lb_batches)).values()),
          'total cluster', Counter(fu.merge_list(lb_batches)).__len__(),
          '\ntotal tweet num ', len(fu.merge_list(tw_batches)))
    ee.stream_semantic_cluster_with_label(tw_batches, lb_batches)
    
    # import TweetKeys
    # fI.remove_files([log_path + file for file in fI.listchildren(log_path, children_type='file')
    #                  if file.endswith('.txt')])
    # for i, _twarr in enumerate(tw_topic_arr):
    #     if not len(_twarr) == 0:
    #         fI.dump_array(log_path + str(i) + '.txt', [tw[TweetKeys.key_cleantext] for tw in _twarr])
    
    # for tw in twarr:
    #     ee.merge_tw_into_cache_back(tw)
    #
    # print('Cluster number', len(ee.cache_back))
    # print('Total tweet number', sum([cache.tweet_number() for cache in ee.cache_back]))
    # topidx = np.argsort([cache.tweet_number() for cache in ee.cache_back])[::-1]
    # print(topidx[0: 30])
    # topvalue = np.array([cache.tweet_number() for cache in ee.cache_back])[topidx]
    # print(topvalue[0: 30])
    #
    # for i in range(len(ee.cache_back)):
    #     if not ee.cache_back[i].tweet_number() > 4:
    #         continue
    #     fI.dump_array(log_path + str(i) + '.txt',
    #                   [dic['tw'][TweetKeys.key_cleantext] for dic in
    #                    ee.cache_back[i].twdict.values()])
    #     fI.dump_array(log_path + str(i) + '.txt',
    #                   [sorted([(k, v) for k, v in
    #                            ee.cache_back[i].keywords.dictionary.items()],
    #                           key=lambda item: item[1]['count'], reverse=True)[:20],
    #                    '----',
    #                    sorted([(k, v) for k, v in
    #                            ee.cache_back[i].entities_non_geo.dictionary.items()],
    #                           key=lambda item: item[1]['count'], reverse=True),
    #                    '----',
    #                    sorted([(k, v) for k, v in
    #                            ee.cache_back[i].entities_geo.dictionary.items()],
    #                           key=lambda item: item[1]['count'], reverse=True),
    #                    '----',
    #                    ee.cache_back[i].tweet_number()
    #                    ], overwrite=False)


def exec_temp(parser):
    # """find tws that classifier fail to recognize on dzs_neg_data, and split them into blocks"""
    # event_extractor = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    # dzs_neg_data = '/home/nfs/cdong/tw/testdata/non_pos_tweets.sum'
    # dzs_neg_twarr = fu.load_array(dzs_neg_data)
    # preds = event_extractor.make_classification(dzs_neg_twarr)
    # false_pos_twarr = [dzs_neg_twarr[idx] for idx, pred in enumerate([pred for pred in preds]) if pred > 0.5]
    # # false_pos_twarr_blocks = au.array_partition(false_pos_twarr, [1] * 10)
    # fu.dump_array('falseevents.txt', false_pos_twarr)
    
    """query for pos events into blocks"""
    data_path = getconfig().summary_path
    log_path = '/home/nfs/cdong/tw/seeding/temp/'
    twarr_blocks = Main2Parser.query_per_query_multi(data_path, parser.seed_query_list)
    print('query over')
    et = EventTrainer()
    et.start_ner_service(pool_size=16, classify=True, pos=True)
    for i in range(len(parser.seed_query_list)):
        twarr_blocks[i] = et.twarr_ner(twarr_blocks[i])
        query_i = parser.seed_query_list[i]
        file_name_i = query_i.all[0].strip('\W') + '_' + '-'.join(query_i.since) + '.sum'
        fu.dump_array(log_path + file_name_i, twarr_blocks[i])
    et.end_ner_service()
    print('ner over, event num queried ', len(twarr_blocks))
    fu.dump_array('events.txt', twarr_blocks, overwrite=False)
    
    # """splitting dzs_neg_data into blocks"""
    # twarr_blocks = list()
    # dzs_neg_data = '/home/nfs/cdong/tw/testdata/non_pos_tweets.sum'
    # dzs_neg_parts = au.array_partition(fu.load_array(dzs_neg_data), (1, 1, 1, 1))
    # for dzs_part in dzs_neg_parts:
    #     twarr_blocks.append(au.random_array_items(dzs_part, 300))
    # my_neg_data = '/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_counter.sum'
    # my_neg_parts = au.array_partition(fu.load_array(my_neg_data), (1, 1, 1, 1))
    # for my_part in my_neg_parts:
    #     twarr_blocks.append(au.random_array_items(my_part, 300))
    #
    # print('tweet distribution ', [len(twarr) for twarr in twarr_blocks], '\n\rtotal cluster', len(twarr_blocks))
    # fu.dump_array('nonevents.txt', twarr_blocks)


def load_clusters_and_labels():
    # event_twarr_blocks = fu.load_array('events.txt')
    # false_pos_twarr_blocks = fu.load_array('falseevents.txt')
    # non_event_twarr_blocks = fu.load_array('nonevents.txt')
    # print('pos event group num:', len(event_twarr_blocks),
    #       'false pos event group num:', len(false_pos_twarr_blocks),
    #       'non event group num:', len(non_event_twarr_blocks))
    # blocks = event_twarr_blocks[0:12] + false_pos_twarr_blocks[::2] + non_event_twarr_blocks[::2]
    # twarr = fu.merge_list(blocks)
    # label = fu.merge_list([[i for _ in range(len(blocks[i]))] for i in range(len(blocks))])
    # return twarr, label
    event_twarr_blocks = fu.load_array('events.txt')
    blocks = event_twarr_blocks[0:12]
    twarr = fu.merge_list(blocks)
    label = fu.merge_list([[i for _ in range(len(blocks[i]))] for i in range(len(blocks))])
    return twarr, label


def create_batches_through_time():
    event_blocks = fu.load_array('events.txt')
    twarr = fu.merge_list(event_blocks)
    label = fu.merge_list([[i for _ in range(len(event_blocks[i]))] for i in range(len(event_blocks))])
    
    idx_time_order = tu.rearrange_idx_by_time(twarr)
    twarr = [twarr[idx] for idx in idx_time_order]
    label = [label[idx] for idx in idx_time_order]
    
    # for idx in range(len(twarr) - 1):
    #     if du.get_timestamp_form_created_at(twarr[idx][TweetKeys.key_created_at].strip()) > \
    #             du.get_timestamp_form_created_at(twarr[idx + 1][TweetKeys.key_created_at].strip()):
    #         raise ValueError('wrong')
    
    idx_parts = au.index_partition(twarr, [1] * int(len(twarr) / 400), random=False)
    tw_batches = [[twarr[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
    lb_batches = [[label[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
    return tw_batches, lb_batches
