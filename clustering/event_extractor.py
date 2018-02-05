from collections import Counter

import numpy as np
import pandas as pd

import utils.array_utils
import utils.function_utils as fu
import utils.multiprocess_utils
import utils.tweet_keys as tk
from seeding.event_classifier import LREventClassifier
from clustering.cluster_service import ClusterService
from utils.id_freq_dict import IdFreqDict
from clustering.gsdmm_semantic_stream import GSDMMSemanticStream
from clustering.gsdpmm_stream import GSDPMMStream
from clustering.gsdpmm_semantic_stream import GSDPMMSemanticStream
from clustering.gsdpmm_stream_retweet import GSDPMMStreamRetweet
from clustering.gsdpmm_stream_ifd_static import GSDPMMStreamIFDStatic
from clustering.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic
from clustering.gsdpmm_stream_semantic_ifd_dynamic import GSDPMMStreamSemanticIFDDynamic


np.random.seed(2333)


class EventExtractor:
    def __init__(self, dict_file, model_file):
        self.freqcounter = self.classifier = self.filteredtw = None
        # self.construct(dict_file, model_file)
    
    def construct(self, dict_file, model_file):
        self.freqcounter = IdFreqDict()
        self.freqcounter.load_dict(dict_file)
        vocab_size = self.freqcounter.vocabulary_size()
        self.classifier = LREventClassifier(vocab_size=vocab_size, learning_rate=0,
                                            unlbreg_lambda=0.01, l2reg_lambda=0.01)
        self.classifier.load_params(model_file)
    
    def make_classification(self, twarr):
        feature_mtx = self.freqcounter.feature_matrix_of_twarr(twarr)
        return self.classifier.predict(feature_mtx)
    
    def filter_twarr(self, twarr, cond=lambda x: x >= 0.5):
        predicts = self.make_classification(twarr)
        return [twarr[idx] for idx, pred in enumerate(predicts) if cond(pred)]
    
    def show_weight_of_words(self):
        thetaE = self.classifier.get_theta()[0]
        table = pd.DataFrame.from_dict(self.freqcounter.worddict.dictionary).T
        table.drop(axis=1, labels=['df'], inplace=True)
        table.sort_values(by='idf', ascending=False, inplace=True)
        for index, series in table.iterrows():
            table.loc[index, 'theta'] = thetaE[0][int(series['id'])]
        print(table)


def stream_cluster_with_label(tw_batches, lb_batches, hold_batch_num, c_type=None):
    # g = GSDMMSemanticStream(hold_batch_num=4)
    # g.set_hyperparams(1, 0.05, 0.05, 0.05, 0.05, 80)
    # g = GSDPMMSemanticStreamStatic(hold_batch_num=4)
    # g = GSDPMMSemanticStreamClusterer(hold_batch_num=4)
    # g.set_hyperparams(900, 0.1, 0.1, 0.1, 0.1)
    
    print('batch size={}, hold batch={}'.format(len(tw_batches[0]), hold_batch_num))
    
    # if c_type is None:
    #     # g = GSDPMMStreamRetweetClusterer(hold_batch_num=4)
    #     # g = GSDPMMStreamClusterer(hold_batch_num=4)
    #     # g = GSDPMMStreamClustererIFDStatic(hold_batch_num=5)
    #     # g = GSDPMMStreamClustererIFDDynamic(hold_batch_num=5)
    #     # g.set_hyperparams(100, 0.01)
    #
    #     g = GSDPMMStreamSemanticDynamic(hold_batch_num=50)
    #     g.set_hyperparams(400, 0.01, 0.03, 0.05, 0.07)
    # else:
    #     g = c_type(hold_batch_num=6)
    #     g.set_hyperparams(400, 0.01)
    
    # g = GSDPMMStreamClusterer(hold_batch_num=hold_batch_num)
    # g.set_hyperparams(100, 0.01)
    # g = GSDPMMStreamIFDDynamic(hold_batch_num=hold_batch_num)
    # g.set_hyperparams(100, 0.01)
    # g = GSDPMMStreamSemanticIFDDynamic(hold_batch_num=hold_batch_num)
    # g.set_hyperparams(400, 0.01, 0.01, 0.01, 0.05)
    g = GSDMMSemanticStream(hold_batch_num=hold_batch_num)
    g.set_hyperparams(0.05, 0.01, 0.01, 0.03, 0.05, 300)
    
    results = z_evo, l_evo, s_evo = list(), list(), list()
    for batch_idx in range(len(tw_batches)):
        print('\r{}\r{}/{} groups, {} tws processed'.format(' ' * 30, batch_idx + 1, len(tw_batches),
              sum([len(t) for t in tw_batches[:batch_idx + 1]])), end='', flush=True)
        z, label = g.input_batch_with_label(tw_batches[batch_idx], lb_batches[batch_idx])
        if not z:
            continue
        if not len(label) == len(z):
            raise ValueError('length inconsistent')
        # if batch_idx > 10:
        # exit(233)
        z_evo.append([int(i) for i in z])
        l_evo.append([int(i) for i in label])
        s_evo.append(g.clusters_similarity())
    
    print(g.get_hyperparams_info())
    fu.dump_array('evolution.txt', [results])
    return results


def analyze_stream(z_evo=None, l_evo=None, s_evo=None, rep_score=0.5):
    """ evaluate the clustering results """
    if not z_evo or not l_evo or not s_evo:
        z_evo, l_evo, s_evo = fu.load_array('evolution.txt')
    
    history_z = set([z for z in utils.array_utils.merge_list(z_evo)])
    print('\nhistory_z {}, effective cluster number {}'.format(sorted(history_z), len(history_z)))
    
    all_labels = utils.array_utils.merge_list(l_evo)
    ne_cluid = int(max(all_labels))
    evo = pd.DataFrame(dtype=int)
    for batch_id in range(len(z_evo)):
        # s_batch is the similarity of every predicted cluster in an iteration
        z_batch, lb_batch, s_batch = z_evo[batch_id], l_evo[batch_id], s_evo[batch_id]
        for k in list(s_batch.keys()):
            if type(k) is str:
                s_batch[int(k)] = s_batch[k]
                s_batch.pop(k)
        df = ClusterService.cluid_label_table([int(i) for i in lb_batch], [int(i) for i in z_batch])
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
    real_event_ids = utils.array_utils.merge_list(l_evo)
    real_event_num = Counter(real_event_ids)
    detected_event_ids = [int(item.split(',')[0]) for item in utils.array_utils.merge_list(evo.values.tolist())
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


def grid_stream_cluster_with_label_multi(tw_batches, lb_batches, c_type=None):
    print('batch num:{}, batch size:{}'.format(len(tw_batches), len(tw_batches[0])))
    
    # a_range = [i for i in range(1, 10, 5)] + [i*10 for i in range(1, 10, 3)] + [i*100 for i in range(1, 10, 2)]
    # b_range = [i/1000 for i in range(1, 10, 2)] + [i/100 for i in range(1, 10, 2)]
    # params = [(a, b) for a in a_range for b in b_range]
    # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
    #                [(tw_batches, lb_batches, GSDPMMStreamClusterer(4), p) for p in params])
    
    # a_range = [100, 10, 1, 0.1]
    # k_range = [30, 50]
    # p_range = c_range = v_range = h_range = [0.01, 0.1, 1]
    # params = [(a, p, c, v, h, k) for a in a_range for p in p_range
    #           for c in c_range for v in v_range for h in h_range for k in k_range]
    # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
    #                 [(tw_batches, lb_batches, SemanticStreamClusterer(4), *p) for p in params])
    
    a_range = [1000, 500, 300, 100, 50]
    b_range = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
    params = [(a, b) for a in a_range for b in b_range]
    if c_type is None:
        # g = GSDPMMStreamClustererIFDStatic(hold_batch_num=4)
        # g = GSDPMMStreamClustererIFDDynamic(hold_batch_num=5)
        # g = GSDPMMStreamClusterer(hold_batch_num=5)
        
        print('grid for GSDPMMStreamSemanticDynamic')
        a_r = [100, 400, 700]
        p_r = c_r = v_r = h_r = [0.01, 0.05, 0.1]
        params = [(a, p, c, v, h) for a in a_r for p in p_r for c in c_r for v in v_r for h in h_r]
        g = GSDPMMStreamSemanticIFDDynamic(hold_batch_num=6)
    else:
        g = c_type(hold_batch_num=6)
    
    res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
                                               [(tw_batches, lb_batches, g, p) for p in params], 10)
    for idx, res in enumerate(sorted(res_list, key=lambda x: x[1], reverse=True)):
        print(res)


def stream_cluster_with_label_multi(tw_batches, lb_batches, clusterer, params):
    from copy import deepcopy
    clusterer = deepcopy(clusterer)
    clusterer.set_hyperparams(*params)
    
    z_evo, l_evo = list(), list()
    for batch_id in range(len(tw_batches)):
        z, label = clusterer.input_batch_with_label(tw_batches[batch_id], lb_batches[batch_id])
        if not z:
            continue
        if not len(label) == len(z):
            raise ValueError('length inconsistent')
        z_evo.append([int(i) for i in z])
        l_evo.append([int(i) for i in label])
    
    """ evaluation """
    all_labels = utils.array_utils.merge_list(l_evo)
    rep_score = 0.7
    non_event_label = int(max(all_labels))
    evolution = pd.DataFrame(dtype=int)
    for batch_id in range(len(z_evo)):
        z_batch, lb_batch = z_evo[batch_id], l_evo[batch_id]
        df = ClusterService.cluid_label_table([int(item) for item in lb_batch],
                                              [int(item) for item in z_batch])
        for batch_id, row in df.iterrows():
            row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
            max_cluid = int(row.index[row_max_idx])
            if row[max_cluid] == 0 or row[max_cluid] < row_sum * rep_score or max_cluid == non_event_label:
                evolution.loc[batch_id, batch_id] = non_event_label
            else:
                evolution.loc[batch_id, batch_id] = max_cluid
    
    evolution.fillna(non_event_label, inplace=True)
    real_event_num = Counter(all_labels)
    detected_event_ids = [int(item) for item in utils.array_utils.merge_list(evolution.values.tolist()) if item != non_event_label]
    detected_event_num = Counter(detected_event_ids)
    detected_event_id = [d for d in sorted(set(detected_event_ids))]
    num_detected = len(detected_event_id)
    event_id_corpus = sorted(set(all_labels).difference({non_event_label}))
    num_corpus = len(event_id_corpus)
    
    # first_col = evolution.columns[0]
    # evolution.loc['eventdistrb', first_col] = '  '.join(
    #     ['{}:{}'.format(eventid, detected_event_num[eventid])
    #      for eventid in sorted(detected_event_num.keys())])
    # evolution.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
    #                                                     for eventid in sorted(real_event_num.keys())])
    # evolution.loc['detectedid', first_col] = str(detected_event_id)
    # evolution.loc['totalevent', first_col] = str(event_id_corpus)
    # evolution.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
    # evolution.fillna('', inplace=True)
    
    info = clusterer.get_hyperparams_info()
    return info, num_detected / num_corpus, num_detected, num_corpus


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('-g', action='store_true', default=False)
    parser.add_argument('-t', nargs='?', default='None')
    args = parser.parse_args()
    
    from clustering.main2clusterer import create_batches_through_time
    tw_batches, lb_batches = create_batches_through_time(batch_size=1000)
    
    type_dict = {'dp': GSDPMMStream,
                 'dss': GSDMMSemanticStream,
                 'st': GSDPMMStreamIFDStatic,
                 'dy': GSDPMMStreamIFDDynamic,
                 'None': None}
    c_type = type_dict[args.t]
    
    if args.g:
        print('grid search')
        grid_stream_cluster_with_label_multi(tw_batches, lb_batches, c_type)
    else:
        print('single test')
        z_evo, l_evo, s_evo = stream_cluster_with_label(tw_batches, lb_batches, 5, c_type)
        analyze_stream(z_evo, l_evo, s_evo)
