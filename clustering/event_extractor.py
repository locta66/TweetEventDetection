from collections import Counter

import numpy as np
import pandas as pd

from clustering.main2clusterer import make_tw_batches, get_tw_batches
import utils.array_utils as au
import utils.function_utils as fu
import utils.tweet_utils as tu
import utils.tweet_keys as tk
import clustering.cluster_service as cs
import utils.timer_utils as tmu
import preprocess.tweet_filter as tflt

from clustering.gsdmm.gsdmm_semantic_stream import GSDMMSemanticStream
from clustering.gsdpmm.gsdpmm_stream import GSDPMMStream
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic
from clustering.gsdpmm.gsdpmm_stream_semantic_ifd_dynamic import GSDPMMStreamSemanticIFDDynamic


np.random.seed(2333)


def stream_cluster_with_label(tw_batches, hold_batch_num):
    print('batch size={}, hold batch={}'.format(len(tw_batches[0]), hold_batch_num))
    g = GSDPMMStreamIFDDynamic()
    g.set_hyperparams(hold_batch_num, 30, 0.01)
    
    """ pre-processing tw batches """
    tmu.check_time()
    tu.twarr_nlp(au.merge_array(tw_batches))
    tmu.check_time(print_func=lambda dt: print('\nspacy over, time elapsed {} s'.format(dt)))
    
    cluid_batches = list()
    for batch_idx, tw_batch in enumerate(tw_batches):
        print('\r{}\r{}/{} batch, {} tws done'.format(
            ' ' * 30, batch_idx + 1, len(tw_batches), sum([len(t) for t in tw_batches[:batch_idx + 1]])),
            end='', flush=True)
        
        cluid_iter = g.input_batch(tw_batch)
        if not cluid_iter:
            continue
        cluid_batches.extend(cluid_iter)
    tmu.check_time(print_func=lambda dt: print('\ncluster over, time elapsed {} s'.format(dt)))
    
    assert len(cluid_batches) == len(tw_batches)
    assert len(au.merge_array(cluid_batches)) == len(au.merge_array(tw_batches))
    fu.dump_array('cluid_evo.txt', cluid_batches)


# def stream_cluster_with_label(tw_batches, lb_batches, hold_batch_num):
#     print('batch size={}, hold batch={}'.format(len(tw_batches[0]), hold_batch_num))
#
#     # g = GSDPMMStreamSemanticIFDDynamic(hold_batch_num=hold_batch_num)
#     # g.set_hyperparams(400, 0.01, 0.01, 0.01, 0.05)
#     g = GSDPMMStreamIFDDynamic(hold_batch_num=hold_batch_num)
#     g.set_hyperparams(30, 0.01)
#
#     results = z_evo, l_evo, s_evo, vs_evo = list(), list(), list(), list()
#     for batch_idx in range(len(tw_batches)):
#         print('\r{}\r{}/{} groups, {} tws processed'.format(' ' * 30, batch_idx + 1, len(tw_batches),
#               sum([len(t) for t in tw_batches[:batch_idx + 1]])), end='', flush=True)
#         z, label = g.input_batch(tw_batches[batch_idx])
#         z = [int(i) for i in z]
#         label = [int(i) for i in label]
#         if not z:
#             continue
#         if not len(label) == len(z):
#             raise ValueError('length inconsistent')
#         z_evo.append(z)
#         l_evo.append(label)
#         s_evo.append(g.cluid2info_by_key('score'))
#         vs_evo.append(g.cluster_mutual_similarity())
#
#         if batch_idx % 50 == 0 and batch_idx >= 60:
#             print('cluster number', len(set(z)))
#             for cluid, score in s_evo[-1].items():
#                 if score <= 0.5:
#                     print('cluid', cluid)
#                     twharr = g.cludict[cluid].get_twharr()
#                     for twh in twharr:
#                         label = '[{}]{}'.format(twh.label, '[x]' if twh.label == 58 else '')
#                         print(label, twh.get(tk.key_text))
#                 print('--')
#             print('\n\n')
#
#     print(g.get_hyperparams_info())
#     fu.dump_array('evolution.txt', list(results))
#     analyze_stream_score_and_vector(*results)


def string_key2int_key(dic):
    for key in list(dic.keys()):
        if type(key) is str:
            dic[int(key)] = dic.pop(key)
    return dic


def analyze_stream_score_and_vector(z_evo=None, l_evo=None, s_evo=None, vs_evo=None, rep_score=0.7):
    """ evaluate the clustering results """
    if not z_evo or not l_evo or not s_evo or not vs_evo:
        z_evo, l_evo, s_evo, vs_evo = fu.load_array('evolution.txt')
        for s in s_evo:
            string_key2int_key(s)
    all_z = set(au.merge_array(z_evo))
    print('\nrecorded cluster number {}'.format(len(all_z)))
    
    all_lb = set(au.merge_array(l_evo))
    ne_lb = int(max(all_lb))
    evo = pd.DataFrame(dtype=int)
    for idx in range(len(z_evo)):
        z_batch, lb_batch, s_batch = z_evo[idx], l_evo[idx], s_evo[idx]
        df = cs.cluid_label_table(z_batch, lb_batch)
        
        for cluid, row in df.iterrows():
            row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
            rep_cluid = int(row.index[row_max_idx])  # which cluster is the representative
            rep_twnum = int(row[rep_cluid])
            is_k_score = round(s_batch[cluid], 3)
            if row.loc[rep_cluid] == 0 or row.loc[rep_cluid] < row_sum * rep_score or rep_cluid == ne_lb:
                fill_cluid = ne_lb
            else:
                fill_cluid = rep_cluid
            evo.loc[idx, cluid] = '{: <4},{:0<5},{}'.format(fill_cluid, is_k_score, rep_twnum)
    
    evo.fillna('', inplace=True)
    effective_cluids = list()
    for info in au.merge_array(evo.values.tolist()):
        if info == '':
            continue
        fill_cluid, is_k_score, rep_twnum = info.split(',')
        fill_cluid = int(fill_cluid)
        if info != '' and fill_cluid != ne_lb:
            effective_cluids.append(fill_cluid)
    detected_e_number = Counter(effective_cluids)
    detected_e_id = sorted(detected_e_number.keys())
    num_detected = len(detected_e_id)
    event_id_corpus = sorted(all_lb.difference({ne_lb}))
    num_corpus = len(event_id_corpus)
    
    first_col = evo.columns[0]
    evo.loc['detectedid', first_col] = str(detected_e_id)
    evo.loc['totalevent', first_col] = str(event_id_corpus)
    evo.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
    evo.to_csv('table.csv')
    print(detected_e_id, num_detected, num_corpus, num_detected / num_corpus)


def analyze_stream(z_evo=None, l_evo=None, s_evo=None, rep_score=0.7):
    """ evaluate the clustering results """
    if not z_evo or not l_evo or not s_evo:
        z_evo, l_evo, s_evo = fu.load_array('evolution.txt')
    
    history_z = set([z for z in au.merge_array(z_evo)])
    print('\nhistory_z {}, effective cluster number {}'.format(sorted(history_z), len(history_z)))
    
    all_labels = au.merge_array(l_evo)
    ne_cluid = int(max(all_labels))
    evo = pd.DataFrame(dtype=int)
    for batch_id in range(len(z_evo)):
        # s_batch is the similarity of every predicted cluster in an iteration
        z_batch, lb_batch, s_batch = z_evo[batch_id], l_evo[batch_id], s_evo[batch_id]
        for k in list(s_batch.keys()):
            if type(k) is str:
                s_batch[int(k)] = s_batch[k]
                s_batch.pop(k)
        df = cs.cluid_label_table(z_batch, lb_batch)
        for pred_cluid, row in df.iterrows():
            row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
            rep_cluid = int(row.index[row_max_idx])
            rep_twnum = int(row[rep_cluid])
            similarity = round(s_batch[pred_cluid], 3)
            if row.loc[rep_cluid] == 0 or row.loc[rep_cluid] < row_sum * rep_score or rep_cluid == ne_cluid:
                evo.loc[batch_id, pred_cluid] = '{: <4},{:0<5},{}'.format(ne_cluid, similarity, rep_twnum)
            else:
                evo.loc[batch_id, pred_cluid] = '{: <4},{:0<3},{}'.format(rep_cluid, similarity, rep_twnum)
    
    evo.fillna('', inplace=True)
    real_event_ids = au.merge_array(l_evo)
    real_event_num = Counter(real_event_ids)
    detected_event_ids = [int(item.split(',')[0]) for item in au.merge_array(evo.values.tolist())
                          if item != '' and int(item.split(',')[0]) != ne_cluid]
    detected_event_num = Counter(detected_event_ids)
    detected_event_id = [d for d in sorted(set(detected_event_ids))]
    num_detected = len(detected_event_id)
    event_id_corpus = sorted(set(all_labels).difference({ne_cluid}))
    num_corpus = len(event_id_corpus)
    
    first_col = evo.columns[0]
    evo.loc['eventdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, detected_event_num[eventid])
                                                   for eventid in sorted(detected_event_num.keys())])
    evo.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
                                                  for eventid in sorted(real_event_num.keys())])
    evo.loc['detectedid', first_col] = str(detected_event_id)
    evo.loc['totalevent', first_col] = str(event_id_corpus)
    evo.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
    evo.fillna('', inplace=True)
    evo.to_csv('table.csv')
    print(detected_event_id, num_detected, num_corpus, num_detected / num_corpus)


# def analyze_stream_k(z_evo=None, l_evo=None, s_evo=None, rep_score=0.7):
#     """ evaluate the clustering results """
#     if not z_evo or not l_evo or not s_evo:
#         z_evo, l_evo, s_evo = fu.load_array('evolution.txt')[0]
#         for s in s_evo:
#             for k in list(s.keys()):
#                 if type(k) is str:
#                     s[int(k)] = s[k]
#                     s.pop(k)
#     all_z = set(au.merge_array(z_evo))
#     # print('\nhistory_z {}, recorded cluster number {}'.format(sorted(all_z), len(all_z)))
#     print('\nrecorded cluster number {}'.format(len(all_z)))
#
#     all_lb = set(au.merge_array(l_evo))
#     ne_lb = int(max(all_lb))
#     evo = pd.DataFrame(dtype=int)
#     for idx in range(len(z_evo)):
#         z_batch, lb_batch, s_batch = z_evo[idx], l_evo[idx], s_evo[idx]
#         df = cs.cluid_label_table(z_batch, lb_batch)
#
#         print('\r{}\r{}'.format(' '*5, idx), end='')
#         # print(sorted(set(df.columns)))
#         # print(sorted(set(s_batch.keys())))
#         diff1 = set(df.index).difference(set(s_batch.keys()))
#         diff2 = set(s_batch.keys()).difference(set(df.index))
#         if len(diff1) > 0 or len(diff2) > 0:
#             print('yashila', diff1, diff2)
#
#         for cluid, row in df.iterrows():
#             row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
#             rep_cluid = int(row.index[row_max_idx])  # which cluster is the representative
#             rep_twnum = int(row[rep_cluid])
#             is_k_score = round(s_batch[cluid], 3)
#             if row.loc[rep_cluid] == 0 or row.loc[rep_cluid] < row_sum * rep_score or rep_cluid == ne_lb:
#                 fill_cluid = ne_lb
#             else:
#                 fill_cluid = rep_cluid
#             evo.loc[idx, cluid] = '{: <4},{:0<5},{}'.format(fill_cluid, is_k_score, rep_twnum)
#             # evo.loc[idx, cluid] = (fill_cluid, is_k_score, rep_twnum)
#
#     evo.fillna('', inplace=True)
#     lb_distrb = Counter(au.merge_array(l_evo))
#     effective_cluids = list()
#     for info in au.merge_array(evo.values.tolist()):
#         if info == '':
#             continue
#         fill_cluid, is_k_score, rep_twnum = info.split(',')
#         fill_cluid = int(fill_cluid)
#         if info != '' and fill_cluid != ne_lb:
#             effective_cluids.append(fill_cluid)
#     detected_e_number = Counter(effective_cluids)
#     detected_e_id = sorted(detected_e_number.keys())
#     num_detected = len(detected_e_id)
#     event_id_corpus = sorted(all_lb.difference({ne_lb}))
#     num_corpus = len(event_id_corpus)
#
#     first_col = evo.columns[0]
#     evo.loc['eventdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, detected_e_number[eventid])
#                                                    for eventid in detected_e_id])
#     evo.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, lb_distrb[eventid])
#                                                   for eventid in sorted(lb_distrb.keys())])
#     evo.loc['detectedid', first_col] = str(detected_e_id)
#     evo.loc['totalevent', first_col] = str(event_id_corpus)
#     evo.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
#     evo.to_csv('table.csv')
#     print(detected_e_id, num_detected, num_corpus, num_detected / num_corpus)


# def grid_stream_cluster_with_label_multi(tw_batches, lb_batches, c_type=None):
#     print('batch num:{}, batch size:{}'.format(len(tw_batches), len(tw_batches[0])))
#
#     # a_range = [i for i in range(1, 10, 5)] + [i*10 for i in range(1, 10, 3)] + [i*100 for i in range(1, 10, 2)]
#     # b_range = [i/1000 for i in range(1, 10, 2)] + [i/100 for i in range(1, 10, 2)]
#     # params = [(a, b) for a in a_range for b in b_range]
#     # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
#     #                [(tw_batches, lb_batches, GSDPMMStreamClusterer(4), p) for p in params])
#
#     # a_range = [100, 10, 1, 0.1]
#     # k_range = [30, 50]
#     # p_range = c_range = v_range = h_range = [0.01, 0.1, 1]
#     # params = [(a, p, c, v, h, k) for a in a_range for p in p_range
#     #           for c in c_range for v in v_range for h in h_range for k in k_range]
#     # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
#     #                 [(tw_batches, lb_batches, SemanticStreamClusterer(4), *p) for p in params])
#
#     a_range = [1000, 500, 300, 100, 50]
#     b_range = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
#     params = [(a, b) for a in a_range for b in b_range]
#     if c_type is None:
#         # g = GSDPMMStreamClustererIFDStatic(hold_batch_num=4)
#         # g = GSDPMMStreamClustererIFDDynamic(hold_batch_num=5)
#         # g = GSDPMMStreamClusterer(hold_batch_num=5)
#
#         print('grid for GSDPMMStreamSemanticDynamic')
#         a_r = [100, 400, 700]
#         p_r = c_r = v_r = h_r = [0.01, 0.05, 0.1]
#         params = [(a, p, c, v, h) for a in a_r for p in p_r for c in c_r for v in v_r for h in h_r]
#         g = GSDPMMStreamSemanticIFDDynamic(hold_batch_num=6)
#     else:
#         g = c_type(hold_batch_num=6)
#
#     mul_func, mul_target = cs.clustering_multi, stream_cluster_with_label_multi
#     res_list = mul_func(mul_target, [(tw_batches, lb_batches, g, p) for p in params], 10)
#     for idx, res in enumerate(sorted(res_list, key=lambda x: x[1], reverse=True)):
#         print(res)
#
#
# def stream_cluster_with_label_multi(tw_batches, lb_batches, clusterer, params):
#     from copy import deepcopy
#     clusterer = deepcopy(clusterer)
#     clusterer.set_hyperparams(*params)
#
#     z_evo, l_evo = list(), list()
#     for batch_id in range(len(tw_batches)):
#         z, label = clusterer.input_batch_with_label(tw_batches[batch_id], lb_batches[batch_id])
#         if not z:
#             continue
#         if not len(label) == len(z):
#             raise ValueError('length inconsistent')
#         z_evo.append([int(i) for i in z])
#         l_evo.append([int(i) for i in label])
#
#     """ evaluation """
#     all_labels = au.merge_array(l_evo)
#     rep_score = 0.7
#     non_event_label = int(max(all_labels))
#     evolution = pd.DataFrame(dtype=int)
#     for batch_id in range(len(z_evo)):
#         z_batch, lb_batch = z_evo[batch_id], l_evo[batch_id]
#         df = cs.cluid_label_table(z_batch, lb_batch)
#         for batch_id, row in df.iterrows():
#             row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
#             max_cluid = int(row.index[row_max_idx])
#             if row[max_cluid] == 0 or row[max_cluid] < row_sum * rep_score or max_cluid == non_event_label:
#                 evolution.loc[batch_id, batch_id] = non_event_label
#             else:
#                 evolution.loc[batch_id, batch_id] = max_cluid
#
#     evolution.fillna(non_event_label, inplace=True)
#     real_event_num = Counter(all_labels)
#     detected_event_ids = [int(item) for item in au.merge_array(evolution.values.tolist()) if
#                           item != non_event_label]
#     detected_event_num = Counter(detected_event_ids)
#     detected_event_id = [d for d in sorted(set(detected_event_ids))]
#     num_detected = len(detected_event_id)
#     event_id_corpus = sorted(set(all_labels).difference({non_event_label}))
#     num_corpus = len(event_id_corpus)
#
#     # first_col = evolution.columns[0]
#     # evolution.loc['eventdistrb', first_col] = '  '.join(
#     #     ['{}:{}'.format(eventid, detected_event_num[eventid])
#     #      for eventid in sorted(detected_event_num.keys())])
#     # evolution.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
#     #                                                     for eventid in sorted(real_event_num.keys())])
#     # evolution.loc['detectedid', first_col] = str(detected_event_id)
#     # evolution.loc['totalevent', first_col] = str(event_id_corpus)
#     # evolution.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
#     # evolution.fillna('', inplace=True)
#
#     info = clusterer.get_hyperparams_info()
#     return info, num_detected / num_corpus, num_detected, num_corpus


batch_size = 100
hold_batch_num = 50


if __name__ == '__main__':
    # TODO 训练关于朝鲜问题的分类器，将分类器用于每轮迭代产生的每个聚类，每个聚类给出代表性标记，
    # 观察分类器能否正确识别出朝鲜问题为代表的聚类；要判别是否有征兆，即观察是否有相应聚类产生
    # 问题是如何训练分类器，这些分类器不能使用过多的训练集数据，可能的话用前一或两个事件的实例作为训练集，
    # 而将最后两或一个事件实例的数据加上负例进行聚类，再用分类器进行判别
    # 可能还需要先在未训练的数据集上进行一定的测试，用于训练的负例可以参考那些所有没有包含Korea的推文，
    # 为此首先应当确定电子所的korea数据在多大程度上包含了这一关键词
    
    import argparse
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('-a', action='store_true', default=False)
    parser.add_argument('-g', action='store_true', default=False)
    parser.add_argument('-t', nargs='?', default='None')
    args = parser.parse_args()
    
    """ read tw data """
    # make_tw_batches(batch_size)
    tw_batches = get_tw_batches()
    
    if args.a:
        print('only analyze')
        # analyze_stream_score_and_vector()
        exit()
    else:
        print('single test')
        stream_cluster_with_label(tw_batches, hold_batch_num)
