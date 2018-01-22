import math

import utils.function_utils as fu
import utils.pattern_utils as pu
import utils.tweet_utils as tu

import numpy as np
import pandas as pd


class ClusterService:
    @staticmethod
    def is_valid_keyword(word):
        return pu.is_valid_keyword(word)
    
    @staticmethod
    def cluster_prediction_table(tw_clu_real, tw_clu_pred, lbl_range=None, pred_range=None):
        """ the rows are predicted cluster id, and the columns are ground truth labels """
        cluster_table = pd.DataFrame(index=sorted(set(tw_clu_pred)) if pred_range is None else pred_range,
                                     columns=sorted(set(tw_clu_real)) if lbl_range is None else lbl_range,
                                     data=0, dtype=int)
        for i in range(len(tw_clu_real)):
            cluster_table.loc[tw_clu_pred[i], tw_clu_real[i]] += 1
        return cluster_table
    
    @staticmethod
    def event_table_recall(tw_cluster_label, tw_cluster_pred, event_cluster_label=None):
        cluster_table = ClusterService.cluster_prediction_table(tw_cluster_label, tw_cluster_pred)
        if event_cluster_label is not None:
            predict_cluster = set([maxid for maxid in np.argmax(cluster_table.values, axis=1)
                                   if maxid in event_cluster_label])
            ground_cluster = set(tw_cluster_label).intersection(set(event_cluster_label))
            cluster_table['PRED'] = [maxid if maxid in event_cluster_label else ''
                                     for maxid in np.argmax(cluster_table.values, axis=1)]
        else:
            predict_cluster = set(np.argmax(cluster_table.values, axis=1))
            ground_cluster = set(tw_cluster_label)
            cluster_table['PRED'] = [maxid for maxid in np.argmax(cluster_table.values, axis=1)]
        recall = len(predict_cluster) / len(ground_cluster)
        return cluster_table, recall, predict_cluster, ground_cluster
    
    @staticmethod
    def create_clusters_with_labels(twarr, label):
        label = [int(i) for i in label]
        if not len(twarr) == len(label):
            raise ValueError('Wrong cluster labels for twarr')
        tw_topic_dict = {}
        for idx in range(len(twarr)):
            tw, lb = twarr[idx], label[idx]
            if lb not in tw_topic_dict:
                tw_topic_dict[lb] = [tw]
            else:
                tw_topic_dict[lb].append(tw)
        return tw_topic_dict
    
    @staticmethod
    def clusters_similarity(twarr, label):
        topic_dict = ClusterService.create_clusters_with_labels(twarr, label)
        return dict([(int(cluid), float(tu.twarr_similarity(_twarr))) for cluid, _twarr in topic_dict.items()])
    
    @staticmethod
    def clustering_multi(func, params, process_num=16):
        param_num = len(params)
        res_list = list()
        for i in range(int(math.ceil(param_num / process_num))):
            res_list += fu.multi_process(func, params[i * process_num: (i + 1) * process_num])
            print('{:<4} / {} params processed'.format(min((i + 1) * process_num, param_num), param_num))
        if not len(res_list) == len(params):
            raise ValueError('Error occur in clustering')
        return res_list
