import __init__

from EventTrainer import EventTrainer
from EventExtractor import EventExtractor
from FunctionUtils import sync_real_time_counter
from ArrayUtils import roc, recall, precision
import Main2Parser
import FileIterator as fI
import TweetKeys


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
    
    pos_twarr = fI.load_array('/home/nfs/cdong/tw/testdata/pos_tweets.sum')
    non_pos_twarr = fI.load_array('/home/nfs/cdong/tw/testdata/non_pos_tweets.sum')
    pos_pred = event_extractor.make_classification(pos_twarr)
    non_pos_pred = event_extractor.make_classification(non_pos_twarr)
    score_label_pairs = [(pred[0], 1) for pred in pos_pred] + [(pred[0], 0) for pred in non_pos_pred]
    print(roc(score_label_pairs))
    print(recall(score_label_pairs, [i/10 for i in range(1, 10)]))
    print(precision(score_label_pairs, [i/10 for i in range(1, 10)]))
    # for idx, pred in enumerate(pos_pred):
    #     if pred < 0.2:
    #         print(pos_twarr[idx][TweetKeys.key_origintext], '\n---\n')


def exec_temp(parser):
    et = EventTrainer()
    ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    data_path = '/home/nfs/cdong/tw/summary/'
    log_path = '/home/nfs/cdong/tw/seeding/temp/'
    
    import os
    if os.path.exists('hehe.txt'):
        twarr = fI.load_array('hehe.txt')
    else:
        et.start_ner_service(pool_size=12, classify=True, pos=True)
        
        # subfiles = [data_path + file for file in fI.listchildren(data_path, children_type='file')
        #             if (file[5: 7] == '08' and file[8: 10] == '29')][0:10]
        # twarr = fI.merge_list([ee.filter_twarr(et.twarr_ner(fI.load_array(file))) for file in subfiles])
        
        file_list = [(data_path + file) for file in fI.listchildren(data_path, children_type='file') if
                     file.endswith('.sum') and parser.is_file_of_query_date(file)]
        print('from', file_list[0], ',', len(file_list), 'files until', '\n', file_list[-1])
        file_list = fI.split_multi_format(file_list, process_num=16)
        twarr_block = fI.multi_process(Main2Parser.query_tw_file_multi,
                                       [(file_list_slice, parser.__class__, parser.query_list, parser.theme,
                                         parser.description) for file_list_slice in file_list])
        twarr = fI.merge_list([et.twarr_ner(twarr) for twarr in twarr_block])
        twarr = Main2Parser.remove_similar_tws(twarr)
        fI.dump_array('hehe.txt', twarr)
        print('Queried', len(twarr), 'tweets,')
        et.end_ner_service()
    
    tw_topic_arr = ee.GSDMM_twarr(twarr)
    
    fI.remove_files([log_path + file for file in fI.listchildren(log_path, children_type='file')])
    for i, _twarr in enumerate(tw_topic_arr):
        if len(_twarr) < 6:
            continue
        fI.dump_array(log_path + str(i) + '.txt',
                      [' '.join([w[0] for w in tw[TweetKeys.key_wordlabels]]) for tw in _twarr])
    
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
