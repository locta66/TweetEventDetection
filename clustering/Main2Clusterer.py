import FileIterator as fI
import FunctionUtils as fu
import Main2Parser
from ArrayUtils import roc_auc
from Configure import getconfig
from EventExtractor import EventExtractor
from EventTrainer import EventTrainer
from FunctionUtils import sync_real_time_counter
from collections import Counter


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
    
    pos_twarr = fI.load_array(getconfig().pos_data_file)
    non_pos_twarr = fI.load_array(getconfig().non_pos_data_file)
    pos_pred = event_extractor.make_classification(pos_twarr)
    non_pos_pred = event_extractor.make_classification(non_pos_twarr)
    scores = [pred[0] for pred in pos_pred] + [pred[0] for pred in non_pos_pred]
    labels = [1 for _ in pos_pred] + [0 for _ in non_pos_pred]
    print(roc_auc(labels, scores))
    # for idx, pred in enumerate(pos_pred):
    #     if pred < 0.2:
    #         print(pos_twarr[idx][TweetKeys.key_origintext], '\n---\n')


def exec_temp(parser):
    et = EventTrainer()
    ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    data_path = '/home/nfs/cdong/tw/summary/'
    log_path = '/home/nfs/cdong/tw/seeding/temp/'
    
    import os
    if os.path.exists('positive.txt'):
        twarr = fu.load_array('positive.txt')
        cluster_label = fu.load_array('label.txt')
    else:
        et.start_ner_service(pool_size=16, classify=True, pos=True)
        
        # subfiles = [data_path + file for file in fI.listchildren(data_path, children_type='file')
        #             if (file[5: 7] == '08' and file[8: 10] == '29')][0:10]
        # twarr = fI.merge_list([ee.filter_twarr(et.twarr_ner(fI.load_array(file))) for file in subfiles])
    
        # file_list = [(data_path + file) for file in fI.listchildren(data_path, children_type='file') if
        #              file.endswith('.sum') and parser.is_file_of_query_date(file)]
        # print('from', file_list[0], ',', len(file_list), 'files until', '\n', file_list[-1])
        # file_list = fI.split_multi_format(file_list, process_num=16)
        # twarr_blocks = fI.multi_process(Main2Parser.query_tw_file_multi,
        #                                [(file_list_slice, parser.__class__, parser.query_list, parser.theme,
        #                                  parser.description) for file_list_slice in file_list])
        # twarr = fI.merge_list([et.twarr_ner(twarr) for twarr in twarr_blocks])
        # print('Queried', len(twarr), 'tweets,', end=' ')
        # twarr = Main2Parser.remove_similar_tws(twarr)
        
        twarr_blocks = Main2Parser.query_per_query_multi(data_path, parser.seed_query_list)
        print('query over')
        for i in range(len(parser.seed_query_list)):
            twarr_blocks[i] = et.twarr_ner(twarr_blocks[i])
            query = parser.seed_query_list[i]
            fu.dump_array(log_path + query.all[0].strip('\W') + '_' + '-'.join(query.since) + '.sum', twarr_blocks[i])
        
        print('ner over')
        twarr = fu.merge_list(twarr_blocks)
        cluster_label = fu.merge_list([[i for _ in twarr_blocks[i]] for i in range(len(twarr_blocks))])
        fu.dump_array('positive.txt', twarr)
        fu.dump_array('label.txt', cluster_label)
        
        et.end_ner_service()
    
    print('Label distribution ', list(dict(Counter(cluster_label)).values()),
          'total cluster', Counter(cluster_label).__len__())
    # twarr = fI.load_array(getconfig().pos_data_file)
    # tw_topic_arr, cluster_pred = ee.GSDMM_twarr_with_label(twarr, cluster_label)
    # tw_topic_arr, cluster_pred = ee.LECM_twarr_with_label(twarr, cluster_label)
    tw_topic_arr, cluster_pred = ee.GSDPMM_twarr_with_label(twarr, cluster_label)
    
    
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
