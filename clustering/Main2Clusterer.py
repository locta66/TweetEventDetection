import __init__

from EventExtractor import EventExtractor
from FunctionUtils import sync_real_time_counter
from ArrayUtils import roc, recall
import Main2Parser
import FileIterator
import TweetKeys

import numpy as np


@sync_real_time_counter('query')
def exec_query(data_path, parser):
    Main2Parser.exec_query(data_path, parser)


def exec_ner(parser):
    Main2Parser.exec_ner(parser)


def exec_classification(seed_parser, test_parser):
    event_extractor = EventExtractor(seed_parser.get_dict_file_name(),
                                     seed_parser.get_param_file_name())
    
    # twarr = FileIterator.load_array(test_parser.get_query_result_file_name())
    
    pos_twarr = FileIterator.load_array('/home/nfs/cdong/tw/testdata/pos_tweets.sum')
    non_pos_twarr = FileIterator.load_array('/home/nfs/cdong/tw/testdata/non_pos_tweets.sum')
    pos_pred = event_extractor.make_classification(pos_twarr)
    non_pos_pred = event_extractor.make_classification(non_pos_twarr)
    score_label_pairs = [(pred[0], 1) for pred in pos_pred] + [(pred[0], 0) for pred in non_pos_pred]
    # print(roc(score_label_pairs))
    # print(recall([(pred[0], 1) for pred in pos_pred], [i/10 for i in range(1, 10)]))
    # print(recall([(pred[0], 0) for pred in non_pos_pred], [i/10 for i in range(1, 10)]))

    for idx, pred in enumerate(pos_pred):
        if pred < 0.2:
            print(pos_twarr[idx][TweetKeys.key_origintext], '\n---\n')

    # for i in [i / 10.0 for i in range(1, 10)]:
    #     count = 0
    #     for pred in prediction:
    #         if pred > i:
    #             count += 1
    #     print(int(i * 100), '%, ', count / len(twarr))


def exec_temp(parser):
    import os
    from EventTrainer import EventTrainer
    et = EventTrainer()
    ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    data_path = '/home/nfs/cdong/tw/summary/'
    log_path = '/home/nfs/cdong/tw/seeding/temp/'
    "----------------------------------------------------------"
    
    if os.path.exists('hehe.txt'):
        twarr = FileIterator.load_array('hehe.txt')
    else:
        et.start_ner_service(pool_size=12, classify=True, pos=True)
        # ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
        # subfiles = [data_path + file for file in FileIterator.listchildren(data_path, children_type='file')
        #             if (file[5: 7] == '08' and file[8: 10] == '29')][0:3]
        # twarr = FileIterator.merge_list([FileIterator.load_array(file) for file in subfiles])
        
        file_list = [(data_path + file) for file in
                     FileIterator.listchildren(data_path, children_type='file') if
                     file.endswith('.sum') and parser.is_file_of_query_date(file)]
        print('from', file_list[0], ',', len(file_list), 'files until', '\n', file_list[-1])
        file_list = FileIterator.split_multi_format(file_list, process_num=16)
        added_twarr_block = FileIterator.multi_process(Main2Parser.query_tw_file_multi,
                                                       [(file_list_slice, parser.__class__,
                                                         parser.query_list, parser.theme,
                                                         parser.description) for file_list_slice in
                                                        file_list])
        twarr = FileIterator.merge_list(added_twarr_block)
        twarr = et.twarr_ner(twarr)[0:10]
        FileIterator.dump_array('hehe.txt', twarr)
        print('Queried', len(twarr), 'tweets,')
        et.end_ner_service()
    
    # for tw in ee.filter_twarr(twarr):
    for tw in twarr:
        ee.merge_tw_into_cache_back(tw)
    
    FileIterator.remove_files(
        [log_path + file for file in FileIterator.listchildren(log_path, children_type='file')])
    print('Cluster number', len(ee.cache_back))
    print('Total tweet number', sum([cache.tweet_number() for cache in ee.cache_back]))
    topidx = np.argsort([cache.tweet_number() for cache in ee.cache_back])[::-1]
    print(topidx[0: 30])
    topvalue = np.array([cache.tweet_number() for cache in ee.cache_back])[topidx]
    print(topvalue[0: 30])
    
    for i in range(len(ee.cache_back)):
        # if not len(ee.cache_back[i].twarr) > 8:
        #     continue
        FileIterator.dump_array(log_path + str(i) + '.txt',
                                [dic['tw'][TweetKeys.key_cleantext] for dic in
                                 ee.cache_back[i].twdict.values()])
        FileIterator.dump_array(log_path + str(i) + '.txt',
                                [sorted([(k, v) for k, v in
                                         ee.cache_back[i].keywords.dictionary.items()],
                                        key=lambda item: item[1]['count'], reverse=True)[:20],
                                 '\n',
                                 sorted([(k, v) for k, v in
                                         ee.cache_back[i].entities_non_geo.dictionary.items()],
                                        key=lambda item: item[1]['count'], reverse=True),
                                 '\n',
                                 sorted([(k, v) for k, v in
                                         ee.cache_back[i].entities_geo.dictionary.items()],
                                        key=lambda item: item[1]['count'], reverse=True), ],
                                overwrite=False)
    "----------------------------------------------------------"
    
    
    # parser.added_twarr = FileIterator.merge_list(added_twarr_block)
    # print('Queried', len(parser.added_twarr), 'tweets,')
    #
    # from Cache import CacheBack
    # twarr = et.twarr_ner(parser.added_twarr)
    # ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    # cache = CacheBack(ee.freqcounter)
    # score_list = list()
    # small_score = 0
    # for tw in twarr:
    #     score = cache.score_with_tw(tw)
    #     print(score)
    #     score_list.append(score)
    #     small_score += 1 if score < 8 else 0
    # print('mean score', np.mean(score_list))
    # print(sorted([(k, v) for k, v in cache.keywords.dictionary.items()][:20],
    #              key=lambda item: item[1]['count'], reverse=True))
    # print(sorted([(k, v) for k, v in cache.entities_non_geo.dictionary.items()],
    #              key=lambda item: item[1]['count'], reverse=True))
    # print(sorted([(k, v) for k, v in cache.entities_geo.dictionary.items()],
    #              key=lambda item: item[1]['count'], reverse=True))
    # print(small_score)
