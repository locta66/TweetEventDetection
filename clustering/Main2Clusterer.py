import math
from collections import Counter

import seeding.main2parser as main2parser
import utils.array_utils
import utils.multiprocess_utils
from config.configure import getcfg
import utils.function_utils as fu
import utils.array_utils as au
import utils.tweet_utils as tu
import utils.tweet_keys as tk
import utils.ark_service_proxy as ark
from clustering.event_extractor import *


@fu.sync_real_time_counter('query')
def exec_query(data_path, parser):
    main2parser.exec_query(data_path, parser)


def exec_ner(parser):
    main2parser.exec_ner(parser)


def exec_classification(seed_parser, test_parser):
    event_extractor = EventExtractor(seed_parser.get_dict_file_name(), seed_parser.get_param_file_name())
    
    # twarr = fI.load_array(test_parser.get_query_result_file_name())
    # pos_pred = event_extractor.make_classification(twarr)
    # print(recall([(pred[0], 1) for pred in pos_pred], [i / 10 for i in range(1, 10)]))
    
    pos_twarr = fu.load_array(getcfg().pos_data_file)
    non_pos_twarr = fu.load_array(getcfg().non_pos_data_file)
    pos_pred = event_extractor.make_classification(pos_twarr)
    non_pos_pred = event_extractor.make_classification(non_pos_twarr)
    scores = [pred[0] for pred in pos_pred] + [pred[0] for pred in non_pos_pred]
    labels = [1 for _ in pos_pred] + [0 for _ in non_pos_pred]
    print(au.score(labels, scores, 'auc'))
    # for idx, pred in enumerate(pos_pred):
    #     if pred < 0.2:
    #         print(pos_twarr[idx][TweetKeys.key_origintext], '\n---\n')


def exec_cluster(parser):
    # twarr, label = load_clusters_and_labels()
    
    # ee = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
    # tw_topic_arr, cluster_pred = ee.LECM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDMM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDPMM_twarr_with_label(twarr, label)
    # tw_topic_arr, cluster_pred = ee.GSDMM_twarr_hashtag_with_label(twarr, label)
    
    tw_batches, lb_batches = create_batches_through_time(batch_size=100)
    # z_evo, l_evo, s_evo = EventExtractor.stream_cluster_with_label(tw_batches, lb_batches)
    # EventExtractor.analyze_stream(z_evo, l_evo, s_evo)
    grid_stream_cluster_with_label_multi(tw_batches, lb_batches)
    
    # twarr, label = load_clusters_and_labels()
    # EventExtractor.batch_cluster_with_label(twarr, label)


def exec_analyze(parser):
    analyze_stream()


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
    data_path = getcfg().summary_path
    log_path = '/home/nfs/cdong/tw/testdata/yli/queried_events_with_keyword/'
    twarr_blocks = main2parser.query_per_query_multi(data_path, parser.seed_query_list)
    print('query done, {} events, {} tws'.format(len(twarr_blocks), sum([len(arr) for arr in twarr_blocks])))
    event_id2info = dict()
    tu.start_ner_service(pool_size=16, classify=True, pos=True)
    for i in range(len(parser.seed_query_list)):
        event_id2info[i] = dict()
        twarr_blocks[i] = tu.twarr_ner(twarr_blocks[i])
        query_i = parser.seed_query_list[i]
        file_name_i = query_i.all[0].strip('\W') + '_' + '-'.join(query_i.since) + '.sum'
        event_id2info[i]['filename'] = file_name_i
        event_id2info[i]['all'] = [w.strip('\W') for w in query_i.all]
        event_id2info[i]['any'] = [w.strip('\W') for w in query_i.any]
        fu.dump_array(log_path + file_name_i, twarr_blocks[i])
    fu.dump_array(log_path + 'event_id2info.txt', [event_id2info])
    tu.end_ner_service()
    print(event_id2info)
    print('ner done')
    fu.dump_array('events.txt', twarr_blocks)
    
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


# def load_clusters_and_labels():
#     event_twarr_blocks = fu.load_array('./events2016.txt')
#     blocks = event_twarr_blocks
#     twarr = fu.merge_list(blocks)
#     label = fu.merge_list([[i for _ in range(len(blocks[i]))] for i in range(len(blocks))])
#
#     label_distrb = Counter(label)
#     print('Topic num:{}, total tw:{}'.format(len(label_distrb), len(twarr)))
#     for idx, cluid in enumerate(sorted(label_distrb.keys())):
#         print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
#     print()
#
#     return twarr, label


def create_batches_through_time(batch_size):
    false_event_twarr = fu.load_array('./data/falseevents.txt')
    # event_blocks = fu.load_array('./data/events.txt')[98:]
    event_blocks = fu.load_array('./data/events.txt')
    event_blocks.append(false_event_twarr)
    
    twarr = utils.array_utils.merge_list(event_blocks)
    label = utils.array_utils.merge_list([[i for _ in range(len(event_blocks[i]))] for i in range(len(event_blocks))])
    
    from preprocess.tweet_filter import twarr_dup_id
    dup_idx_list = set(twarr_dup_id(twarr))
    twarr = [twarr[idx] for idx in range(len(twarr)) if idx not in dup_idx_list]
    label = [label[idx] for idx in range(len(label)) if idx not in dup_idx_list]
    
    label_distrb = Counter(label)
    print('Topic num:{}, total tw:{}'.format(len(label_distrb), len(twarr)))
    for idx, cluid in enumerate(sorted(label_distrb.keys())):
        print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
    print()
    
    idx_time_order = tu.rearrange_idx_by_time(twarr)
    twarr = [twarr[idx] for idx in idx_time_order]
    label = [label[idx] for idx in idx_time_order]
    
    # for idx in range(len(twarr) - 1):
    #     if du.get_timestamp_form_created_at(twarr[idx][TweetKeys.key_created_at].strip()) > \
    #             du.get_timestamp_form_created_at(twarr[idx + 1][TweetKeys.key_created_at].strip()):
    #         raise ValueError('wrong')
    
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
    
    idx_rearrange = random_idx_for_item(label, {len(event_blocks) - 1})
    twarr = [twarr[idx] for idx in idx_rearrange]
    label = [label[idx] for idx in idx_rearrange]
    """ cutting twarr and label into batches of batch_size """
    # idx_parts = au.index_partition(twarr, [1] * int(len(twarr) / batch_size), random=False)
    # tw_batches = [[twarr[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
    # lb_batches = [[label[j] for j in idx_parts[i]] for i in range(len(idx_parts))]
    
    full_idx = [i for i in range(len(twarr))]
    batch_num = int(math.ceil(len(twarr) / batch_size))
    tw_batches = [[twarr[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
    lb_batches = [[label[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
    
    # label = fu.merge_list(lb_batches)
    # print(len(twarr), len(label))
    # label_distrb = Counter(label)
    # print('\n\nTopic num:{}, total tw:{}'.format(len(label_distrb), len(twarr)))
    # for idx, cluid in enumerate(sorted(label_distrb.keys())):
    #     print('{:<3}:{:<6}'.format(cluid, label_distrb[cluid]), end='\n' if (idx + 1) % 10 == 0 else '')
    # print()
    
    return tw_batches, lb_batches


if __name__ == '__main__':
    # files = ['events2012.txt', ]
    # tu.start_ner_service(16)
    # for file in files:
    #     blocks = fu.load_array(file)
    #     for _twarr in blocks:
    #         # ark.twarr_ark(_twarr) if tk.key_ark not in _twarr[0] else None
    #         tu.twarr_ner(_twarr) if tk.key_wordlabels not in _twarr[0] else None
    #     print(sorted([('id' + str(idx), len(_twarr)) for idx, _twarr in enumerate(blocks)], key=lambda x: x[1]))
    #     print('file:{}'.format(file))
    #     fu.dump_array(file, blocks)
    
    import sklearn
    import re
    from sklearn import svm
    import utils.array_utils as au
    str_arr = fu.load_array('sim_info.txt')
    feature = list()
    labels = list()
    for string in str_arr:
        num_arr = [float(s) for s in re.findall('\d\.\d+|\d+', string)]
        if num_arr[4] < 0.5:
            continue
        feature.append([num_arr[1], num_arr[3], num_arr[4]])
        labels.append(1 if num_arr[0] == num_arr[2] else 0)
        print(num_arr, feature[-1], labels[-1])
    
    split_idx = int(len(feature) * 0.5)
    trainX, testX = feature[split_idx:], feature[:split_idx]
    trainY, testY = labels[split_idx:], labels[:split_idx]
    
    clf = svm.SVC()
    clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    auc = sklearn.metrics.roc_auc_score(testY, predY)
    print(auc)
    for idx in range(len(predY)):
        print(predY[idx], testY[idx])
    # all_pred = clf.predict(feature)
    # print(sum(all_pred), len(feature))
    # print('--pred--')
    # wrong = 0
    # for idx in range(len(predY)):
    #     p = predY[idx]
    #     l = labels[idx]
    #     if not p == l:
    #         print(feature[idx], p, l)
    #         wrong += 1
    # print(len(feature), wrong)
        # print("pred {}, true {}".format(p, l))
    # print(clf.predict(feature))
