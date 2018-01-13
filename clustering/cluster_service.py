import math

import utils.function_utils as fu
import utils.pattern_utils as pu

import numpy as np
import pandas as pd


class ClusterService:
    @staticmethod
    def is_valid_keyword(word):
        return pu.is_valid_keyword(word)
    
    @staticmethod
    def cluster_label_prediction_table(tw_clu_real, tw_clu_pred, lbl_range=None, pred_range=None):
        """ the rows are predicted cluster id, and the columns are ground truth labels """
        cluster_table = pd.DataFrame(index=set(tw_clu_pred) if pred_range is None else pred_range,
                                     columns=set(tw_clu_real) if lbl_range is None else lbl_range,
                                     data=0, dtype=int)
        for i in range(len(tw_clu_real)):
            row = tw_clu_pred[i]
            col = tw_clu_real[i]
            cluster_table.loc[row, col] += 1
        return cluster_table
    
    # @staticmethod
    # def print_cluster_prediction_table(tw_cluster_label, tw_cluster_pred):
    #     print(ClusterService.cluster_label_prediction_table(tw_cluster_label, tw_cluster_pred))
    
    @staticmethod
    def event_table_recall(tw_cluster_label, tw_cluster_pred, event_cluster_label=None):
        cluster_table = ClusterService.cluster_label_prediction_table(tw_cluster_label, tw_cluster_pred)
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
    def create_clusters_with_labels(twarr, tw_cluster_label):
        if not len(twarr) == len(tw_cluster_label):
            raise ValueError('Wrong cluster labels for twarr')
        tw_topic_arr = [[] for _ in range(max(tw_cluster_label) + 1)]
        for d in range(len(tw_cluster_label)):
            tw_topic_arr[tw_cluster_label[d]].append(twarr[d])
        return tw_topic_arr
    
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
