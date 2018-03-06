import math
from collections import Counter

import utils.array_utils as au
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.tweet_utils as tu
import utils.tweet_keys as tk
# import classifying.k.classifier as kc

import numpy as np
import pandas as pd


def is_valid_keyword(word):
    return pu.is_valid_keyword(word)


def clustering_multi(func, params, process_num=20):
    param_num = len(params)
    res_list = list()
    for i in range(int(math.ceil(param_num / process_num))):
        res_list += mu.multi_process(func, params[i * process_num: (i + 1) * process_num])
        print('{:<4} / {} params processed'.format(min((i + 1) * process_num, param_num), param_num))
    if not len(res_list) == len(params):
        raise ValueError('Error occur in clustering')
    return res_list


def cluid_label_table(cluidarr, labelarr, cluid_range=None, label_range=None):
    """ the rows are predicted cluster id, and the columns are ground truth labels """
    cluid_range = sorted(set(cluidarr)) if cluid_range is None else cluid_range
    label_range = sorted(set(labelarr)) if label_range is None else label_range
    cluster_table = pd.DataFrame(index=cluid_range, columns=label_range, data=0, dtype=int)
    pair_counter = Counter([(int(cluidarr[i]), int(labelarr[i])) for i in range(len(labelarr))])
    for pair, count in pair_counter.items():
        cluster_table.loc[pair[0], pair[1]] = count
    return cluster_table


def create_clusters_with_labels(twarr, cluidarr):
    if not len(twarr) == len(cluidarr):
        raise ValueError('len(twarr)={} inconsistent with len(cluidar)={}'.format(len(twarr), len(cluidarr)))
    cluid2twarr = dict()
    for idx in range(len(twarr)):
        tw, cluid = twarr[idx], cluidarr[idx]
        if cluid not in cluid2twarr:
            cluid2twarr[cluid] = [tw]
        else:
            cluid2twarr[cluid].append(tw)
    return cluid2twarr


def create_clusters_and_vectors(twarr, cluidarr):
    cluid2twarr = create_clusters_with_labels(twarr, cluidarr)
    cluid2vector = dict()
    for cluid, clutwarr in cluid2twarr.items():
        cluid2vector[cluid] = tu.twarr_vector_info(clutwarr, 'vector')
    cords = sorted(set(cluidarr))
    sim_matrix = pd.DataFrame(index=cords, columns=cords, data=0.0)
    for i in range(len(cords)):
        sim_matrix.loc[i, i] = 0.0
        for j in range(i + 1, len(cords)):
            v1, v2 = cluid2vector[i].reshape([1, -1]), cluid2vector[j].reshape([1, -1])
            sim = 0 if (v1 is None or v2 is None) else au.cosine_similarity(v1, v2)
            sim_matrix.loc[i, j] = sim_matrix.loc[j, i] = sim
    return cluid2vector, sim_matrix


def cluster_inner_similarity(twarr, cluidarr):
    cluid2twarr = create_clusters_with_labels(twarr, cluidarr)
    cluid2sim = dict()
    for cluid, clutwarr in cluid2twarr.items():
        cluid2sim[cluid] = float(tu.twarr_vector_info(clutwarr, 'similarity'))
    return cluid2sim


# def clusters_classify_k(twarr, cluidarr):
#     cluid2twarr = create_clusters_with_labels(twarr, cluidarr)
#     cluid2kscore = dict()
#     threshold = 0.5
#     for cluid, clutwarr in cluid2twarr.items():
#         preds = kc.predict([tw.get(tk.key_text) for tw in clutwarr], threshold=threshold)
#         score = np.mean(preds)
#         cluid2kscore[cluid] = score
#     return cluid2kscore
