import Levenshtein
from sklearn import metrics

from preprocess.tweet_filter import filter_twarr
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_utils as tu
import utils.tweet_keys as tk
from config.configure import getcfg
from seeding.event_trainer import EventTrainer


query_process_num = 16


@fu.sync_real_time_counter('query')
def exec_query(data_path, parser):
    print('seeding query')
    data_path = fi.add_sep_if_needed(data_path)
    subfiles = fi.listchildren(data_path, children_type='file')
    file_list = [(data_path + file_name) for file_name in subfiles if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    if not file_list:
        parser.added_twarr = list()
        return
    print(file_list[0], '\n', len(file_list), 'files until', '\n', file_list[-1])
    file_list = fu.split_multi_format(file_list, process_num=query_process_num)
    added_twarr_block = fu.multi_process(query_tw_file_multi,
                                         [(file_list_slice, parser.__class__, parser.query_list,
                                           parser.theme, parser.description)
                                          for file_list_slice in file_list])
    parser.added_twarr = fu.merge_list(added_twarr_block)
    print('Queried', len(parser.added_twarr), 'tweets,')
    remove_similar_tws(parser.added_twarr)
    print(len(parser.added_twarr), 'accepted.\n')
    # for tw in parser.added_twarr:
    #     print(tw[TweetKeys.key_origintext], '\n---\n')
    # file_name = parser.get_query_result_file_name()
    file_name = '/home/nfs/cdong/tw/testdata/yying/queried/NaturalDisaster.sum'
    fu.dump_array(file_name, parser.added_twarr)
    print(file_name, 'written.\n')
    exec_ner(file_name)


@fu.sync_real_time_counter('unlabelled')
def exec_query_unlabelled(data_path, parser):
    data_path = fi.add_sep_if_needed(data_path)
    file_list = [file_name for file_name in fi.listchildren(data_path, children_type='file') if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    date_group = au.group_array_by_condition(file_list, lambda item: item[:11])
    file_list = [[(data_path + file_name) for file_name in group] for group in date_group]
    print('total file number', sum([len(f_list) for f_list in file_list]))
    start = 0
    process_num = query_process_num
    twarr_block = list()
    while True:
        added_twarr_block = fu.multi_process(query_tw_file_multi,
                                             [(file_slice, parser.__class__, parser.query_list,
                                               parser.theme, parser.description) for file_slice in
                                              file_list[start * process_num: (start + 1) * process_num]])
        twarr_block.extend(added_twarr_block)
        if (start + 1) * process_num > len(file_list):
            break
        start += 1
    # One day per block
    print('Queried', sum([len(twarr) for twarr in twarr_block]), 'tweets.')
    for twarr in twarr_block:
        remove_similar_tws(twarr)
    tu.start_ner_service(pool_size=16)
    for twarr in twarr_block:
        tu.twarr_ner(twarr)
        EventTrainer.extract_tw_with_high_freq_entity(twarr)
    print(sum([len(twarr) for twarr in twarr_block]), 'tweets with high freq entities.')
    # twarr = remove_similar_tws(fI.merge_list(twarr_block))
    twarr = fu.merge_list(twarr_block)
    print(len(twarr), 'accepted.')
    fu.dump_array(parser.get_query_result_file_name() + '1', twarr)
    tu.end_ner_service()


@fu.sync_real_time_counter('counter')
def exec_query_counter(data_path, parser):
    data_path = fi.add_sep_if_needed(data_path)
    file_list = [(data_path + file_name) for file_name in
                 fi.listchildren(data_path, children_type='file') if
                 file_name.endswith('.sum') and parser.is_file_of_query_date(file_name)]
    print(file_list[0], '\n', len(file_list), 'files until', '\n', file_list[-1])
    file_list = fu.split_multi_format(file_list, process_num=query_process_num)
    added_twarr_block = fu.multi_process(query_tw_file_multi,
                                         [(file_list_slice, parser.__class__, parser.query_list,
                                           parser.theme,
                                           parser.description) for file_list_slice in file_list])
    parser.added_twarr = fu.merge_list(added_twarr_block)
    print('Queried', len(parser.added_twarr), 'tweets,')
    parser.added_twarr = au.random_array_items(parser.added_twarr, 40000)
    # remove_similar_tws(parser.added_twarr)
    # file_name = parser.get_query_result_file_name()
    file_name = '/home/nfs/cdong/tw/testdata/yying/queried/MyNegative.sum'
    fu.dump_array(file_name, parser.added_twarr)
    print(len(parser.added_twarr), 'accepted.\n', file_name, 'written.\n')
    exec_ner(file_name)


def query_tw_file_multi(file_list, parser_class, query_list, theme, description):
    # Query tweets that satisfies condition of parser from a given file set
    parser = parser_class(query_list, theme, description)
    for file in file_list:
        parser.read_tweet_from_json_file(file)
    return parser.added_twarr


# def query_tw_files_in_path(json_path, *args, **kwargs):
#     parser = kwargs['parser']
#     subfiles = fI.listchildren(json_path, children_type='file')
#     for subfile in subfiles:
#         if not subfile.endswith('.sum'):
#             continue
#         json_file = json_path + subfile
#         parser.read_tweet_from_json_file(json_file)


def query_per_query_multi(data_path, query_list):
    """make query and return corresponding twarr in the order of queries"""
    return fu.multi_process(func=per_query, args_list=[(data_path, query) for query in query_list])


def per_query(data_path, query):
    subfiles = fi.listchildren(data_path, children_type='file')
    file_list = [(data_path + file_name) for file_name in subfiles if file_name.endswith('.sum') and
                 query.is_time_desired(query.time_of_tweet(file_name, source='filename'))]
    for file in file_list:
        twarr = fu.load_array(file)
        for tw in twarr:
            query.append_desired_tweet(tw, usingtwtime=False)
    return query.query_results


def remove_similar_tws(twarr):
    for i in range(len(twarr) - 1, -1, -1):
        for j in range(len(twarr) - 1, i, -1):
            istr = twarr[i][tk.key_text].lower()
            jstr = twarr[j][tk.key_text].lower()
            dist = Levenshtein.distance(istr, jstr) + 1
            if max(len(istr), len(jstr)) / dist >= 5:
                del twarr[j]
    return twarr


def exec_ner(file_name):
    tu.start_ner_service()
    twarr = fu.load_array(file_name)
    twarr = tu.twarr_ner(twarr)
    fu.dump_array(file_name, twarr)
    tu.end_ner_service()


def exec_train_with_outer(seed_parser, unlb_parser, cntr_parser):
    et = EventTrainer()
    seed_twarr = fu.load_array(seed_parser.get_query_result_file_name())
    unlb_twarr = fu.load_array(unlb_parser.get_query_result_file_name())
    cntr_twarr = fu.load_array(cntr_parser.get_query_result_file_name())
    
    dzs_pos_twarr = fu.load_array(getcfg().pos_data_file)
    dzs_non_pos_twarr = fu.load_array(getcfg().non_pos_data_file)
    dzs_pos_train, dzs_pos_test = au.array_partition(dzs_pos_twarr, (1, 1))
    dzs_non_pos_train, dzs_non_pos_test = au.array_partition(dzs_non_pos_twarr, (1, 1))
    pos_twarr = seed_twarr + dzs_pos_train * 5
    neg_twarr = cntr_twarr + dzs_non_pos_train
    
    localcounter, event_classifier = et.train_and_test(
        pos_twarr, unlb_twarr, neg_twarr, dzs_pos_test, dzs_non_pos_test)
    
    pos_pred = event_classifier.predict(localcounter.feature_matrix_of_twarr(dzs_pos_test))
    non_pos_pred = event_classifier.predict(localcounter.feature_matrix_of_twarr(dzs_non_pos_test))
    lebels = [1 for _ in pos_pred] + [0 for _ in non_pos_pred]
    scores = [pred[0] for pred in pos_pred] + [pred[0] for pred in non_pos_pred]
    
    print('\nTest on dianzisuo')
    print('auc', metrics.roc_auc_score(lebels, scores))
    precision, recall, thresholds = metrics.precision_recall_curve(lebels, scores)
    
    last_idx = 0
    for ref in [i / 10 for i in range(3, 8)]:
        for idx in range(last_idx, len(thresholds)):
            if thresholds[idx] >= ref:
                print('threshold', round(thresholds[idx], 2), '\tprecision', round(precision[idx], 5),
                      '\trecall', round(recall[idx], 5))
                last_idx = idx
                break
    
    localcounter.dump_worddict(seed_parser.get_dict_file_name())
    event_classifier.save_params(seed_parser.get_param_file_name())


def temp(parser):
    efe = EventTrainer()
    # efe.start_ner_service(pool_size=16)
    # twarr = fI.load_array('/home/nfs/cdong/tw/summary/2016_06_11_14.sum')
    # efe.twarr_ner(twarr)
    # for tw in twarr:
    #     print(tw[TweetKeys.key_origintext], '\n--------------------')
    #     print(tw[TweetKeys.key_cleantext], '\n--------------------')
    #     print(tw[TweetKeys.key_wordlabels], '\n')
    
    # import numpy as np
    # twarr = fI.load_array('/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_unlabelled.sum')
    # fI.dump_array('/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_unlabelled.origin', twarr)
    # while True:
    #     for i in range(len(twarr)-1, -1, -1):
    #         length = len(twarr)
    #         if length <= 2500:
    #             fI.dump_array('/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_unlabelled.sum', twarr)
    #             return
    #         if 1/length >= np.random.random():
    #             del twarr[i]


def construct_feature_matrix(seed_parser, unlb_parser, cntr_parser):
    print('construct_feature_matrix')
    et = EventTrainer()
    
    seed_twarr = fu.load_array(seed_parser.get_query_result_file_name())
    print('seed_twarr:', len(seed_twarr))
    unlb_twarr = fu.load_array(unlb_parser.get_query_result_file_name())
    cntr_twarr = fu.load_array(cntr_parser.get_query_result_file_name())
    print('cntr_twarr:', len(cntr_twarr))
    
    dzs_pos_twarr = fu.load_array(getcfg().pos_data_file)
    dzs_neg_twarr = fu.load_array(getcfg().non_pos_data_file)
    print('dzs_pos_twarr:', len(cntr_twarr))
    dzs_pos_train, dzs_pos_test = au.array_partition(dzs_pos_twarr, (1, 1))
    print('dzs_pos_train:', len(dzs_pos_train))
    print('dzs_pos_test:', len(dzs_pos_test))
    dzs_neg_train, dzs_neg_test = au.array_partition(dzs_neg_twarr, (1, 1))
    print('dzs_neg_train:', len(dzs_neg_train))
    print('dzs_neg_test:', len(dzs_neg_test))
    
    localcounter = et.train_and_test(seed_twarr + dzs_pos_train, unlb_twarr,
                                     cntr_twarr + dzs_neg_train,
                                     dzs_pos_test, dzs_neg_test)
    print('vocabulary_size', localcounter.vocabulary_size())

    # import numpy as np
    from scipy import sparse, io
    def create_matrix_and_dump(_localcounter, twarr, filename):
        mtx = _localcounter.feature_matrix_of_twarr(twarr)
        print(len(mtx), type(mtx))
        print(len(mtx[0]), type(mtx[0]))
        contract_mtx = sparse.csr_matrix(mtx)
        file = getcfg().dc_test + filename
        io.mmwrite(file, contract_mtx, field='real')
        # read_contract_mtx = io.mmread(file)
        # dense_mtx = read_contract_mtx.todense()
        # print(np.sum(np.matrix(mtx) - np.matrix(dense_mtx)))
    
    # def read_matrix(filename):
    #     from scipy import io
    #     mtx_from_file = io.mmread(filename)
    #     return mtx_from_file.todense()
    
    create_matrix_and_dump(localcounter, seed_twarr, 'my_pos.txt')
    create_matrix_and_dump(localcounter, cntr_twarr, 'my_neg.txt')
    
    create_matrix_and_dump(localcounter, dzs_pos_train, 'dzs_pos_train.txt')
    create_matrix_and_dump(localcounter, dzs_neg_train, 'dzs_neg_train.txt')
    create_matrix_and_dump(localcounter, dzs_pos_test, 'dzs_pos_test.txt')
    create_matrix_and_dump(localcounter, dzs_neg_test, 'dzs_neg_test.txt')


def exec_pre_test(test_data_path):
    subfiles = fi.listchildren(test_data_path, children_type='file')
    # file_list = fu.split_multi_format(
    #     [(test_data_path + file) for file in subfiles if file.endswith('.json')], process_num=6)
    # twarr_blocks = fu.multi_process(fi.summary_unzipped_tweets_multi,
    #                                 [(file_list_slice,) for file_list_slice in file_list])
    twarr_blocks = filter_twarr([fu.load_array(file) for file in subfiles if file.endswith('.json')])
    twarr = fu.merge_list(twarr_blocks)
    
    tu.start_ner_service(pool_size=16)
    tu.twarr_ner(twarr)
    tu.end_ner_service()
    
    all_ids = set(fu.load_array(test_data_path + 'test_ids_all.csv'))
    pos_ids = set(fu.load_array(test_data_path + 'test_ids_pos.csv'))
    non_pos_ids = all_ids.difference(pos_ids)
    
    pos_twarr = list()
    non_pos_twarr = list()
    for tw in twarr:
        twid = tw[tk.key_id]
        if twid in pos_ids:
            pos_twarr.append(tw)
        elif twid in non_pos_ids:
            non_pos_twarr.append(tw)
    
    fu.dump_array(getcfg().pos_data_file, pos_twarr)
    fu.dump_array(getcfg().non_pos_data_file, non_pos_twarr)
    # fu.dump_array('/home/nfs/cdong/tw/testdata/yying/queried/pos_tweets.sum', pos_twarr)
    # fu.dump_array('/home/nfs/cdong/tw/testdata/yying/queried/non_pos_tweets.sum', non_pos_twarr)
